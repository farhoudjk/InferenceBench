"""
vLLM instance launcher.

Manages subprocess lifecycle for vLLM OpenAI-compatible API servers.
Each strategy requires its own vLLM instance with a dedicated port.
Supports sequential launch (one strategy at a time) to avoid GPU contention.
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
from dataclasses import dataclass
from typing import Optional

import aiohttp

from config.strategies import Strategy

logger = logging.getLogger(__name__)


def _log_system_state(label: str, port: int) -> None:
    """Dump port, process, and GPU state to help diagnose strategy-transition failures."""
    lines = [f"=== System state [{label}] ==="]

    # Port state
    try:
        result = subprocess.run(["ss", "-tlnp", f"sport = :{port}"], capture_output=True, text=True)
        port_line = next((l for l in result.stdout.splitlines() if f":{port}" in l), None)
        lines.append(f"  Port {port}: {'BOUND — ' + port_line.strip() if port_line else 'FREE'}")
    except Exception as e:
        lines.append(f"  Port check error: {e}")

    # All vLLM-related processes
    try:
        result = subprocess.run(["pgrep", "-a", "-f", "vllm"], capture_output=True, text=True)
        procs = result.stdout.strip().splitlines()
        if procs:
            lines.append(f"  vLLM processes ({len(procs)}):")
            for p in procs[:5]:
                lines.append(f"    {p[:120]}")
        else:
            lines.append("  vLLM processes: none")
    except Exception as e:
        lines.append(f"  Process check error: {e}")

    # pt_main_thread (vLLM engine worker)
    try:
        result = subprocess.run(["pgrep", "-a", "-f", "pt_main_thread"], capture_output=True, text=True)
        procs = result.stdout.strip().splitlines()
        if procs:
            lines.append(f"  pt_main_thread workers: {len(procs)}")
            for p in procs[:3]:
                lines.append(f"    {p[:120]}")
    except Exception:
        pass

    # GPU memory via nvidia-smi
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

BASE_PORT = 8100
VLLM_STARTUP_TIMEOUT_S = 180       # large models take time to load
HEALTH_POLL_INTERVAL_S  = 3.0


@dataclass
class VLLMInstance:
    strategy: Strategy
    model: str
    port: int
    process: Optional[subprocess.Popen] = None
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


class VLLMLauncher:
    """
    Sequential launcher: starts one vLLM instance, runs benchmark, shuts it down.
    Avoids multi-instance GPU contention during strategy comparison.
    """

    def __init__(
        self,
        model: str,
        results_dir: str = "results",
        extra_vllm_args: Optional[list[str]] = None,
        gpu_indices: Optional[list[int]] = None,
    ):
        self.model = model
        self.results_dir = results_dir
        self.extra_vllm_args = extra_vllm_args or []
        self.gpu_indices = gpu_indices or []
        self._current_instance: Optional[VLLMInstance] = None

    def _build_command(self, strategy: Strategy, port: int) -> list[str]:
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--port", str(port),
            "--host", "127.0.0.1",
        ]
        cmd += strategy.to_vllm_args()
        cmd += self.extra_vllm_args
        return cmd

    def _launch_process(self, strategy: Strategy, port: int) -> subprocess.Popen:
        cmd = self._build_command(strategy, port)
        log_path = os.path.join(self.results_dir, f"vllm_{strategy.name}.log")
        os.makedirs(self.results_dir, exist_ok=True)

        logger.info(f"Launching vLLM [{strategy.name}] on port {port}")
        logger.info(f"Python:  {sys.executable}")
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"Log:     {log_path}")

        env = os.environ.copy()
        # Triton flash attention uses MMA kernels that require Ampere+ (compute 8.0+).
        # V100 is Volta (compute 7.0) — disable to fall back to xformers/standard attention.
        env["VLLM_USE_TRITON_FLASH_ATTN"] = "0"

        # Pin vLLM to the GPU(s) chosen in the TUI/CLI. Without this, CUDA hands the
        # subprocess physical GPU 0 by default regardless of the selection, so a busy
        # GPU 0 causes CUDA OOM even when another GPU is free. vLLM sees the exposed
        # GPUs renumbered from 0, so its tensor-parallel workers map onto exactly the
        # selected devices. We set this only on the subprocess env (not os.environ),
        # so the parent's diagnostics and the NVML monitor keep physical indexing.
        if self.gpu_indices:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in self.gpu_indices)

        if strategy.tensor_parallel_size > 1:
            # Multi-GPU TP workers run on this same host and rendezvous via Gloo/NCCL.
            # With multiple NICs present (docker0, tailscale0, etc.) PyTorch's default
            # interface selection can pick one that drops the loopback connection used
            # by vLLM's in_the_same_node_as() Gloo barrier, hanging until timeout.
            # Pin both backends to loopback so rendezvous stays local.
            env["GLOO_SOCKET_IFNAME"] = "lo"
            env["NCCL_SOCKET_IFNAME"] = "lo"

            # This host's two V100s each sit behind their own separate AMD PCIe
            # host bridge (no shared switch, no NVLink — confirmed via `nvidia-smi
            # topo -m` showing PHB). CUDA reports P2P as capable, but real P2P
            # DMA across two host bridges hangs NCCL indefinitely (GPU spins at
            # 100% SM with no progress). Disabling P2P forces NCCL to use the
            # SHM/socket transport instead, which works reliably on this topology.
            env["NCCL_P2P_DISABLE"] = "1"

        # Some vLLM code paths (e.g. fp8 quantization calibration) probe for extra
        # repo files over HTTP even when the safetensors checkpoint is already cached.
        # vLLM 0.6 uses multiprocessing spawn for its engine process, which re-reads
        # os.environ from the system environment — not the env dict passed to Popen.
        # So we must inject the token into the current process's os.environ as well,
        # so it propagates through the full spawn chain.
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

        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,   # process group for clean teardown
                env=env,
            )
        return process

    async def _wait_for_health(self, instance: VLLMInstance) -> bool:
        """Poll /health until vLLM frontend is up, then warmup-verify the engine."""
        deadline = time.time() + VLLM_STARTUP_TIMEOUT_S
        last_log = time.time()
        poll_count = 0
        t0 = getattr(instance, "_t0", time.time())
        logger.info(f"Waiting for vLLM [{instance.strategy.name}] (timeout: {VLLM_STARTUP_TIMEOUT_S}s) ...")

        async with aiohttp.ClientSession() as session:

            # Phase 1: wait for /health to return 200
            while time.time() < deadline:
                if instance.process and instance.process.poll() is not None:
                    logger.error(
                        f"vLLM [{instance.strategy.name}] process exited early "
                        f"(returncode={instance.process.returncode})"
                    )
                    _log_system_state(f"process-exited:{instance.strategy.name}", instance.port)
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
                        f"vLLM [{instance.strategy.name}] still starting ... "
                        f"({elapsed:.0f}s elapsed, {remaining:.0f}s remaining)"
                    )
                    last_log = time.time()

                await asyncio.sleep(HEALTH_POLL_INTERVAL_S)
            else:
                logger.error(
                    f"vLLM [{instance.strategy.name}] failed to start within {VLLM_STARTUP_TIMEOUT_S}s"
                )
                return False

            elapsed = time.time() - t0
            logger.info(
                f"vLLM [{instance.strategy.name}] /health OK after {poll_count} polls "
                f"({elapsed:.1f}s). Waiting for engine to accept requests..."
            )

            # Phase 2: warmup loop — retry until engine serves a real completion
            instance._warmup_deadline = deadline
            warmup_ok = await self._warmup_request(session, instance)

        if not warmup_ok:
            logger.error(
                f"vLLM [{instance.strategy.name}] engine never became ready within {VLLM_STARTUP_TIMEOUT_S}s"
            )
            _log_system_state(f"warmup-failed:{instance.strategy.name}", instance.port)
            return False

        instance.startup_time_s = time.time() - t0
        logger.info(
            f"vLLM [{instance.strategy.name}] engine verified ready ({instance.startup_time_s:.1f}s total)"
        )
        return True

    async def _warmup_request(self, session: aiohttp.ClientSession, instance: VLLMInstance) -> bool:
        """Send a minimal completion to verify the engine (not just frontend) is serving."""
        payload = {
            "model": instance.model,
            "prompt": "Hi",
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
        # Retry warmup until it succeeds or we hit the outer deadline
        deadline = getattr(instance, "_warmup_deadline", time.time() + 120)
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            # Check if process died between attempts
            if instance.process and instance.process.poll() is not None:
                logger.error(
                    f"Warmup: vLLM process died during warmup attempts "
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

    async def start(self, strategy: Strategy, port: int = BASE_PORT) -> VLLMInstance:
        """Launch vLLM for the given strategy and wait until healthy."""
        _log_system_state(f"before-start:{strategy.name}", port)
        # Evict any stale port-holder or GPU-resident engine worker from a prior run
        self._kill_port_holder(port)
        self._kill_gpu_workers()
        t0 = time.time()
        process = self._launch_process(strategy, port)
        instance = VLLMInstance(strategy=strategy, model=self.model, port=port, process=process)
        instance._t0 = t0  # store for _wait_for_health

        # Poll for readiness
        healthy = await self._wait_for_health(instance)
        if instance.startup_time_s == 0.0:
            instance.startup_time_s = time.time() - t0

        if not healthy:
            self.stop(instance)
            raise RuntimeError(f"vLLM instance [{strategy.name}] failed to start.")

        self._current_instance = instance
        return instance

    def stop(self, instance: Optional[VLLMInstance] = None) -> None:
        """Terminate the vLLM process and all child processes, then wait for port to free."""
        target = instance or self._current_instance
        if target is None:
            return

        port = target.port
        logger.info(f"Stopping vLLM [{target.strategy.name}] (port {port})")
        _log_system_state(f"before-stop:{target.strategy.name}", port)

        # Step 1: kill the process group we launched
        if target.process is not None:
            try:
                pgid = os.getpgid(target.process.pid)
                os.killpg(pgid, signal.SIGTERM)
                target.process.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(target.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            target.process = None

        # Step 2: kill any process still holding the port (vLLM RPC workers escape the group)
        self._kill_port_holder(port)

        # Step 3: kill any remaining GPU-resident Python processes from this vLLM run
        # (the engine worker spawned via multiprocessing survives killpg on the frontend)
        self._kill_gpu_workers()

        target.process = None
        self._current_instance = None

        # Step 4: wait until port is actually free
        deadline = time.time() + 30
        while time.time() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    break
            time.sleep(1)
        else:
            # One more hard kill attempt then give up
            self._kill_port_holder(port)
            time.sleep(2)

        time.sleep(2)  # extra buffer for GPU memory release
        _log_system_state(f"after-stop:{target.strategy.name}", port)
        logger.info(f"vLLM [{target.strategy.name}] stopped.")

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

    @staticmethod
    def _kill_gpu_workers() -> None:
        """Instance-callable wrapper around the module-level cleanup function."""
        kill_stale_gpu_workers()

    async def run_with_strategy(self, strategy: Strategy, coro, port: int = BASE_PORT):
        """
        Context helper: launch strategy → run coroutine → teardown.
        Usage:
            results = await launcher.run_with_strategy(strategy, benchmark_fn(instance))
        """
        instance = await self.start(strategy, port)
        try:
            return await coro(instance)
        finally:
            self.stop(instance)


def kill_stale_gpu_workers() -> None:
    """SIGKILL any stale vLLM processes: both server frontends and GPU engine workers.

    Two sources of stale processes:
    1. vLLM API server frontends (vllm.entrypoints.openai.api_server) — these are
       the parent processes that may survive from a previous TUI session. Even if
       their GPU worker died, the frontend can respawn a new worker and re-occupy
       the GPU when the next run starts.
    2. GPU engine workers (pt_main_thread) — spawned via multiprocessing, escape
       the parent process group, and hold all GPU memory indefinitely after a crash.
    """
    killed: list[str] = []

    # ── Kill vLLM server frontends ───────────────────────────────────────────
    try:
        result = subprocess.run(
            ["pgrep", "-f", "vllm.entrypoints.openai.api_server"],
            capture_output=True, text=True,
        )
        own_pid = os.getpid()
        for line in result.stdout.strip().splitlines():
            try:
                pid = int(line.strip())
            except ValueError:
                continue
            if pid == own_pid:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(f"vLLM server PID {pid}")
            except ProcessLookupError:
                pass
    except Exception as e:
        logger.debug(f"kill_stale_gpu_workers (server sweep): {e}")

    # ── Kill GPU-resident engine workers ─────────────────────────────────────
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        for line in result.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 2:
                continue
            pid_str, mem_str = parts
            try:
                pid = int(pid_str)
                mem_mib = int(mem_str)
            except ValueError:
                continue
            if mem_mib < 1000:
                continue
            try:
                with open(f"/proc/{pid}/comm") as f:
                    comm = f.read().strip()
            except OSError:
                continue
            if "python" not in comm.lower() and "pt_main" not in comm.lower():
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(f"GPU worker PID {pid} ({mem_mib} MiB, comm={comm})")
            except ProcessLookupError:
                pass
    except Exception as e:
        logger.debug(f"kill_stale_gpu_workers (GPU sweep): {e}")

    for entry in killed:
        logger.info(f"Killed stale {entry}")


async def get_vllm_metrics(instance: VLLMInstance) -> dict:
    """
    Fetch Prometheus metrics from vLLM /metrics endpoint.
    Returns parsed key metrics relevant to benchmark analysis.
    """
    target_metrics = {
        "vllm:gpu_cache_usage_perc",
        "vllm:num_running_seqs",
        "vllm:num_waiting_seqs",
        "vllm:num_preemptions_total",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:spec_decode_draft_acceptance_rate",
        "vllm:spec_decode_efficiency",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_emitted_tokens_total",
    }

    parsed = {}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                instance.metrics_url, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status != 200:
                    return parsed
                text = await resp.text()

        for line in text.splitlines():
            if line.startswith("#"):
                continue
            for key in target_metrics:
                if line.startswith(key):
                    parts = line.rsplit(" ", 1)
                    if len(parts) == 2:
                        try:
                            parsed[key] = float(parts[1])
                        except ValueError:
                            pass
    except Exception:
        pass

    return parsed
