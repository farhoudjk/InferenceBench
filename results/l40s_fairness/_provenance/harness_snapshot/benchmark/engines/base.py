"""
Engine abstraction: pluggable serving-backend lifecycle + metrics.

Each concrete Engine (VLLMEngine, SGLangEngine, ...) owns everything specific
to its CLI flags, hardware workarounds, and scheduler-metrics endpoint. This
base class owns the lifecycle shape that's identical across engines: spawn a
subprocess, poll /health, warmup-verify with a real completion, SIGTERM-then-
SIGKILL process-group teardown, wait for the port to free.
"""

import asyncio
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Protocol

import aiohttp

from config.strategies import Strategy

logger = logging.getLogger(__name__)

STARTUP_TIMEOUT_S = 500       # large models / slow CUDA graph warmup
HEALTH_POLL_INTERVAL_S = 3.0


class ProcessHandle(Protocol):
    """Minimal interface _wait_for_health()/stop() need from a running
    engine instance, regardless of whether it's a plain subprocess.Popen
    (vLLM, SGLang) or a Docker container (TGI, TensorRT-LLM's Triton image).
    """

    def poll(self) -> Optional[int]:
        """None if still running, else the exit code."""
        ...

    def terminate(self) -> None:
        """Ask the process/container to shut down gracefully (SIGTERM-equivalent)."""
        ...

    def kill(self) -> None:
        """Force-kill the process/container (SIGKILL-equivalent)."""
        ...

    def wait(self, timeout: Optional[float] = None) -> int:
        """Block until exit, raising subprocess.TimeoutExpired on timeout."""
        ...


class PopenHandle:
    """ProcessHandle wrapping a plain subprocess.Popen — the vLLM/SGLang case.
    Terminate/kill target the whole process group (preexec_fn=os.setsid at
    launch), since vLLM's engine worker escapes the immediate child process."""

    def __init__(self, popen: subprocess.Popen):
        self._popen = popen

    @property
    def pid(self) -> int:
        return self._popen.pid

    @property
    def returncode(self) -> Optional[int]:
        return self._popen.returncode

    def poll(self) -> Optional[int]:
        return self._popen.poll()

    def terminate(self) -> None:
        try:
            os.killpg(os.getpgid(self._popen.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

    def kill(self) -> None:
        try:
            os.killpg(os.getpgid(self._popen.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    def wait(self, timeout: Optional[float] = None) -> int:
        return self._popen.wait(timeout=timeout)


@dataclass
class EngineInstance:
    engine_name: str
    strategy: Strategy
    model: str
    port: int
    process: Optional[ProcessHandle] = None
    base_url: str = ""
    startup_time_s: float = 0.0

    def __post_init__(self):
        self.base_url = f"http://localhost:{self.port}"

    @property
    def completions_url(self) -> str:
        return f"{self.base_url}/v1/completions"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"

    @property
    def metrics_url(self) -> str:
        return f"{self.base_url}/metrics"   # Prometheus endpoint


@dataclass
class ServerMetrics:
    """Normalized scheduler-metrics schema, common across engines.

    Fields are Optional — an engine that doesn't expose an equivalent metric
    leaves it None rather than a fake 0.0, so "not measured" stays
    distinguishable from "measured zero" downstream in requests.csv.
    """
    kv_cache_usage_pct: Optional[float] = None
    num_waiting_seqs: Optional[int] = None
    num_running_seqs: Optional[int] = None
    num_preemptions_total: Optional[int] = None
    spec_draft_acceptance_rate: Optional[float] = None
    spec_decode_efficiency: Optional[float] = None
    prefix_cache_hit_rate: Optional[float] = None  # prefix/KV-cache reuse rate
    raw: dict = field(default_factory=dict)  # unmapped engine-specific metrics, for debugging

    def as_dict(self) -> dict:
        return {
            "kv_cache_usage_pct": self.kv_cache_usage_pct,
            "num_waiting_seqs": self.num_waiting_seqs,
            "num_running_seqs": self.num_running_seqs,
            "num_preemptions_total": self.num_preemptions_total,
            "spec_draft_acceptance_rate": self.spec_draft_acceptance_rate,
            "spec_decode_efficiency": self.spec_decode_efficiency,
            "prefix_cache_hit_rate": self.prefix_cache_hit_rate,
        }


def _log_system_state(label: str, port: int, process_patterns: list[str]) -> None:
    """Dump port, process, and GPU state to help diagnose strategy-transition failures."""
    lines = [f"=== System state [{label}] ==="]

    try:
        result = subprocess.run(["ss", "-tlnp", f"sport = :{port}"], capture_output=True, text=True)
        port_line = next((l for l in result.stdout.splitlines() if f":{port}" in l), None)
        lines.append(f"  Port {port}: {'BOUND — ' + port_line.strip() if port_line else 'FREE'}")
    except Exception as e:
        lines.append(f"  Port check error: {e}")

    for pattern in process_patterns:
        try:
            result = subprocess.run(["pgrep", "-a", "-f", pattern], capture_output=True, text=True)
            procs = result.stdout.strip().splitlines()
            if procs:
                lines.append(f"  '{pattern}' processes ({len(procs)}):")
                for p in procs[:5]:
                    lines.append(f"    {p[:120]}")
        except Exception as e:
            lines.append(f"  Process check error ({pattern}): {e}")

    try:
        result = subprocess.run(["pgrep", "-a", "-f", "pt_main_thread"], capture_output=True, text=True)
        procs = result.stdout.strip().splitlines()
        if procs:
            lines.append(f"  pt_main_thread workers: {len(procs)}")
            for p in procs[:3]:
                lines.append(f"    {p[:120]}")
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        for row in result.stdout.strip().splitlines():
            parts = [x.strip() for x in row.split(",")]
            if len(parts) == 4:
                idx, used, free, total = parts
                lines.append(f"  GPU {idx}: {used} MiB used / {total} MiB total ({free} MiB free)")
    except Exception as e:
        lines.append(f"  GPU check error: {e}")

    logger.info("\n".join(lines))


class Engine(ABC):
    """Pluggable serving-backend adapter: CLI args, lifecycle, metrics."""

    name: str = ""  # set by subclass, e.g. "vllm" | "sglang"

    def __init__(
        self,
        model: str,
        results_dir: str = "results",
        extra_args: Optional[list[str]] = None,
    ):
        self.model = model
        self.results_dir = results_dir
        self.extra_args = extra_args or []
        self._current_instance: Optional[EngineInstance] = None

    # ── Engine-specific seams (implemented by subclasses) ────────────────

    def build_args(self, strategy: Strategy, port: int) -> list[str]:
        """Full CLI arg list (argv[1:], including --model/--port/--host in
        whatever form this engine spells them) to launch this engine's server.

        Only required if this engine launches via `python -m module_name()`
        (the default _launch_cmd()). Engines that override _launch_cmd()
        directly (e.g. a console-script entrypoint) don't need this."""
        raise NotImplementedError

    def module_name(self) -> str:
        """Module to run via `sys.executable -m <module_name>`.

        Only required if this engine uses the default _launch_cmd()."""
        raise NotImplementedError

    def python_executable(self) -> str:
        """Interpreter to launch this engine's server with.

        Defaults to the current process's interpreter. Override when an
        engine's package can't coexist in the same venv as the others (e.g.
        pinned to an incompatible torch/CUDA build) and instead lives in its
        own venv — return that venv's python path here."""
        return sys.executable

    def _launch_cmd(self, strategy: Strategy, port: int) -> list[str]:
        """Full argv (including interpreter/entrypoint) to launch this
        engine's server. Default: `<python_executable()> -m <module_name()>
        <build_args()>` — override for engines invoked via a console-script
        entrypoint (e.g. `trtllm-serve serve <model> ...`) instead of a
        `python -m` module."""
        return [self.python_executable(), "-m", self.module_name()] + self.build_args(strategy, port)

    def env_overrides(self, strategy: Strategy) -> dict[str, str]:
        """Engine+hardware-specific env vars layered onto the subprocess env."""
        return {}

    def preflight_check(self, strategy: Strategy) -> None:
        """Raise RuntimeError for strategy/hardware combos known not to work."""
        return None

    @classmethod
    def process_match_patterns(cls) -> list[str]:
        """pgrep -f patterns identifying this engine's frontend process.

        Classmethod (not instance method): kill_stale_gpu_workers() needs
        these patterns from the engine registry at startup/cleanup time,
        before any Engine is instantiated with a model/results_dir.
        """
        return []

    @abstractmethod
    async def get_server_metrics(self, instance: EngineInstance) -> ServerMetrics:
        """Scrape this engine's metrics endpoint into the normalized schema."""

    async def finalize_metrics(self, instance: EngineInstance) -> Optional[ServerMetrics]:
        """Called once by run_benchmark() right after the request loop
        finishes (server still running, before engine.stop() tears it down).

        Most engines don't need this — get_server_metrics() already polls
        live during the run. It exists for engines whose metrics endpoint
        can't be scraped mid-run (see TensorRTLLMEngine's override) and
        would otherwise silently end up with only-null scheduler/cache
        fields in every request row.
        """
        return None

    # ── Shared process lifecycle ──────────────────────────────────────────

    def _log_path(self, strategy: Strategy) -> str:
        log_suffix = f"_seqs{strategy.max_num_seqs}" if strategy.max_num_seqs is not None else ""
        os.makedirs(self.results_dir, exist_ok=True)
        return os.path.join(self.results_dir, f"{self.name}_{strategy.name}{log_suffix}.log")

    def _subprocess_env(self, strategy: Strategy) -> dict[str, str]:
        """Build the env dict for a launched subprocess: current env +
        engine-specific overrides + cached HF token propagation.

        Some engines' code paths probe for extra repo files over HTTP even when
        the checkpoint is already cached. Engines that spawn subprocesses via
        multiprocessing re-read os.environ from the system environment, not the
        env dict passed to Popen — so also inject into the current process's
        os.environ so it propagates through the full spawn chain.
        """
        env = os.environ.copy()
        env.update(self.env_overrides(strategy))

        if "HF_TOKEN" not in env and "HUGGING_FACE_HUB_TOKEN" not in env:
            try:
                from huggingface_hub import HfFolder
                token = HfFolder.get_token()
                if token:
                    env["HF_TOKEN"] = token
                    env["HUGGING_FACE_HUB_TOKEN"] = token
                    os.environ.setdefault("HF_TOKEN", token)
                    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
            except Exception as e:
                logger.debug(f"Could not read cached HF token: {e}")
        return env

    def _launch_process(self, strategy: Strategy, port: int) -> ProcessHandle:
        """Launch this engine as a plain subprocess (the default: vLLM,
        SGLang). Container-based engines (e.g. TGI's Docker image) override
        this entirely rather than reuse subprocess.Popen — see
        docker_engine.py's DockerContainerHandle for that shape."""
        cmd = self._launch_cmd(strategy, port) + self.extra_args
        log_path = self._log_path(strategy)

        logger.info(f"Launching {self.name} [{strategy.name}] on port {port}")
        logger.info(f"Python:  {self.python_executable()}")
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"Log:     {log_path}")

        env = self._subprocess_env(strategy)

        with open(log_path, "w") as log_file:
            popen = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,   # process group for clean teardown
                env=env,
            )
        return PopenHandle(popen)

    async def _wait_for_health(self, instance: EngineInstance) -> bool:
        """Poll /health until the frontend is up, then warmup-verify the engine."""
        deadline = time.time() + STARTUP_TIMEOUT_S
        last_log = time.time()
        poll_count = 0
        t0 = getattr(instance, "_t0", time.time())
        logger.info(f"Waiting for {self.name} [{instance.strategy.name}] (timeout: {STARTUP_TIMEOUT_S}s) ...")

        async with aiohttp.ClientSession() as session:

            while time.time() < deadline:
                if instance.process and instance.process.poll() is not None:
                    logger.error(
                        f"{self.name} [{instance.strategy.name}] process exited early "
                        f"(returncode={instance.process.returncode})"
                    )
                    _log_system_state(f"process-exited:{instance.strategy.name}", instance.port,
                                       self.process_match_patterns())
                    return False
                try:
                    async with session.get(
                        instance.health_url, timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        poll_count += 1
                        if resp.status == 200:
                            break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass

                if time.time() - last_log >= 30:
                    elapsed = time.time() - t0
                    remaining = deadline - time.time()
                    logger.info(
                        f"{self.name} [{instance.strategy.name}] still starting ... "
                        f"({elapsed:.0f}s elapsed, {remaining:.0f}s remaining)"
                    )
                    last_log = time.time()

                await asyncio.sleep(HEALTH_POLL_INTERVAL_S)
            else:
                logger.error(
                    f"{self.name} [{instance.strategy.name}] failed to start within {STARTUP_TIMEOUT_S}s"
                )
                return False

            elapsed = time.time() - t0
            logger.info(
                f"{self.name} [{instance.strategy.name}] /health OK after {poll_count} polls "
                f"({elapsed:.1f}s). Waiting for engine to accept requests..."
            )

            instance._warmup_deadline = deadline
            warmup_ok = await self._warmup_request(session, instance)

        if not warmup_ok:
            logger.error(
                f"{self.name} [{instance.strategy.name}] engine never became ready within {STARTUP_TIMEOUT_S}s"
            )
            _log_system_state(f"warmup-failed:{instance.strategy.name}", instance.port,
                               self.process_match_patterns())
            return False

        instance.startup_time_s = time.time() - t0
        logger.info(
            f"{self.name} [{instance.strategy.name}] engine verified ready ({instance.startup_time_s:.1f}s total)"
        )
        return True

    async def _warmup_request(self, session: aiohttp.ClientSession, instance: EngineInstance) -> bool:
        """Send a minimal completion to verify the engine (not just frontend) is serving."""
        payload = {
            "model": instance.model,
            "prompt": "Hi",
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
        deadline = getattr(instance, "_warmup_deadline", time.time() + 120)
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            if instance.process and instance.process.poll() is not None:
                logger.error(
                    f"Warmup: {self.name} process died during warmup attempts "
                    f"(returncode={instance.process.returncode})"
                )
                return False
            try:
                async with session.post(
                    instance.completions_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10, sock_connect=5),
                ) as resp:
                    if resp.status == 200:
                        logger.info(f"Warmup succeeded on attempt {attempt}")
                        return True
                    body = await resp.text()
                    logger.info(f"Warmup attempt {attempt}: HTTP {resp.status}: {body[:300]}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.info(f"Warmup attempt {attempt}: {type(e).__name__}: {e}")
            await asyncio.sleep(3)
        return False

    async def start(self, strategy: Strategy, port: int) -> EngineInstance:
        """Launch this engine for the given strategy and wait until healthy."""
        self.preflight_check(strategy)
        _log_system_state(f"before-start:{strategy.name}", port, self.process_match_patterns())
        self._kill_port_holder(port)
        self._kill_gpu_workers()
        t0 = time.time()
        process = self._launch_process(strategy, port)
        instance = EngineInstance(engine_name=self.name, strategy=strategy, model=self.model,
                                   port=port, process=process)
        instance._t0 = t0

        healthy = await self._wait_for_health(instance)
        if instance.startup_time_s == 0.0:
            instance.startup_time_s = time.time() - t0

        if not healthy:
            self.stop(instance)
            raise RuntimeError(f"{self.name} instance [{strategy.name}] failed to start.")

        self._current_instance = instance
        return instance

    def stop(self, instance: Optional[EngineInstance] = None) -> None:
        """Terminate the process and all child processes, then wait for port to free."""
        target = instance or self._current_instance
        if target is None:
            return

        port = target.port
        logger.info(f"Stopping {self.name} [{target.strategy.name}] (port {port})")
        _log_system_state(f"before-stop:{target.strategy.name}", port, self.process_match_patterns())

        if target.process is not None:
            target.process.terminate()
            try:
                target.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                target.process.kill()
            target.process = None

        self._kill_port_holder(port)
        self._kill_gpu_workers()

        target.process = None
        self._current_instance = None

        deadline = time.time() + 30
        while time.time() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    break
            time.sleep(1)
        else:
            self._kill_port_holder(port)
            time.sleep(2)

        time.sleep(2)  # extra buffer for GPU memory release
        _log_system_state(f"after-stop:{target.strategy.name}", port, self.process_match_patterns())
        logger.info(f"{self.name} [{target.strategy.name}] stopped.")

    @staticmethod
    def _kill_port_holder(port: int) -> None:
        """SIGKILL any process listening on the given TCP port."""
        try:
            result = subprocess.run(
                ["ss", "-tlnp", f"sport = :{port}"],
                capture_output=True, text=True,
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "pid=" in line:
                    for pid_str in re.findall(r"pid=(\d+)", line):
                        try:
                            os.kill(int(pid_str), signal.SIGKILL)
                            logger.info(f"Killed port-{port} holder PID {pid_str}")
                        except ProcessLookupError:
                            pass
        except Exception as e:
            logger.debug(f"_kill_port_holder({port}): {e}")

    def _kill_gpu_workers(self) -> None:
        from benchmark.engines import kill_stale_gpu_workers
        kill_stale_gpu_workers()

    async def run_with_strategy(self, strategy: Strategy, coro, port: int):
        """Context helper: launch strategy → run coroutine → teardown."""
        instance = await self.start(strategy, port)
        try:
            return await coro(instance)
        finally:
            self.stop(instance)
