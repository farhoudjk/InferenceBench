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
VLLM_STARTUP_TIMEOUT_S = 500       # large models / slow CUDA graph warmup on V100
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
        # Physical GPU(s) this launcher owns. Pins the vLLM subprocess to them via
        # CUDA_VISIBLE_DEVICES and scopes stale-process cleanup, so a second job on
        # another GPU can run in parallel without the two clobbering each other.
        self.gpu_indices = gpu_indices if gpu_indices is not None else [0]
        self._current_instance: Optional[VLLMInstance] = None

    # Models whose CUDA graph warmup hangs on V100 (Volta, compute 7.0).
    # These are typically bfloat16-native models vLLM casts to float16, which
    # causes graph capture to stall indefinitely. --enforce-eager bypasses it.
    _ENFORCE_EAGER_MODELS = {
        "mistral-nemo", "gemma-2", "gemma-3", "phi-4",
        "qwq", "deepseek-r1",
    }

    # Models whose native context (max_position_embeddings) is shorter than
    # the strategy default of 8192 — vLLM refuses to start otherwise
    # ("max_model_len is greater than the derived max_model_len").
    _MAX_MODEL_LEN_OVERRIDES = {
        "llama-2": 4096,
    }

    def _needs_enforce_eager(self) -> bool:
        model_lower = self.model.lower()
        return any(k in model_lower for k in self._ENFORCE_EAGER_MODELS)

    def _native_max_model_len(self) -> Optional[int]:
        model_lower = self.model.lower()
        for key, limit in self._MAX_MODEL_LEN_OVERRIDES.items():
            if key in model_lower:
                return limit
        return None

    # AWQ kernels require Turing or newer (compute capability >= 7.5).
    # V100 (Volta) is 7.0 — vLLM builds and starts loading the model before
    # discovering this, wasting ~15-25s per attempt with a deep traceback.
    _AWQ_MIN_COMPUTE_CAPABILITY = (7, 5)

    def _check_quantization_support(self, strategy: Strategy) -> None:
        is_awq = "awq" in self.model.lower() or strategy.quantization == "awq"
        if not is_awq:
            return
        import torch
        if not torch.cuda.is_available():
            return
        major, minor = torch.cuda.get_device_capability(0)
        if (major, minor) < self._AWQ_MIN_COMPUTE_CAPABILITY:
            gpu_name = torch.cuda.get_device_properties(0).name
            raise RuntimeError(
                f"Model '{self.model}' requires AWQ quantization, which needs GPU compute "
                f"capability >= {self._AWQ_MIN_COMPUTE_CAPABILITY[0]}.{self._AWQ_MIN_COMPUTE_CAPABILITY[1]} "
                f"(Turing or newer). This GPU ({gpu_name}) is compute {major}.{minor} and does not "
                f"support AWQ kernels — this is a hardware limitation, not fixable via configuration. "
                f"Use an unquantized or fp8/gptq checkpoint instead."
            )

    # Chunked prefill's prefix-attention path (PagedAttention.forward_prefix →
    # context_attention_fwd, a Triton kernel in vllm/attention/ops/prefix_prefill.py)
    # emits MMA→MMA layout conversions that Triton only codegens for Ampere+
    # (compute capability >= 8.0). On Volta (V100, 7.0) the server boots fine but the
    # first chunked forward pass aborts the worker with
    # "mma -> mma layout conversion is only supported on Ampere". The XFormers
    # fallback only covers pure-prefill/pure-decode; any chunked step with a cached
    # prefix goes through the Triton kernel regardless, so there is no config escape.
    _CHUNKED_PREFILL_MIN_COMPUTE_CAPABILITY = (8, 0)

    def _check_chunked_prefill_support(self, strategy: Strategy) -> None:
        if not strategy.enable_chunked_prefill:
            return
        import torch
        if not torch.cuda.is_available():
            return
        major, minor = torch.cuda.get_device_capability(0)
        if (major, minor) < self._CHUNKED_PREFILL_MIN_COMPUTE_CAPABILITY:
            gpu_name = torch.cuda.get_device_properties(0).name
            raise RuntimeError(
                f"Strategy '{strategy.name}' enables chunked prefill, whose prefix-attention "
                f"Triton kernel needs GPU compute capability >= "
                f"{self._CHUNKED_PREFILL_MIN_COMPUTE_CAPABILITY[0]}."
                f"{self._CHUNKED_PREFILL_MIN_COMPUTE_CAPABILITY[1]} (Ampere or newer). "
                f"This GPU ({gpu_name}) is compute {major}.{minor}; the server would start but "
                f"the first chunked forward pass aborts with an 'mma -> mma layout conversion is "
                f"only supported on Ampere' Triton assertion. This is a hardware limitation, not "
                f"fixable via configuration. Skip the chunked_* strategies (and the chunked_prefill "
                f"dimension) on this host, or run them on an Ampere+ GPU."
            )

    def _build_command(self, strategy: Strategy, port: int) -> list[str]:
        native_limit = self._native_max_model_len()
        if native_limit is not None and strategy.max_model_len > native_limit:
            logger.info(
                f"Capping max_model_len for model {self.model}: "
                f"{strategy.max_model_len} → {native_limit} (model's native context limit)"
            )
            strategy.max_model_len = native_limit

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--port", str(port),
            "--host", "127.0.0.1",
        ]
        cmd += strategy.to_vllm_args()
        # Some models' CUDA graph warmup hangs on V100; fall back to eager execution.
        if self._needs_enforce_eager() and "--enforce-eager" not in cmd:
            cmd += ["--enforce-eager"]
            logger.info(f"Auto-adding --enforce-eager for model {self.model} (V100 CUDA graph workaround)")
        cmd += self.extra_vllm_args
        return cmd

    def _launch_process(self, strategy: Strategy, port: int) -> subprocess.Popen:
        cmd = self._build_command(strategy, port)
        # Include max_num_seqs in the filename so a max_num_seqs sweep (same
        # strategy, different concurrency) doesn't have each run overwrite the
        # previous run's log — only the last sweep value's log would survive.
        log_suffix = f"_seqs{strategy.max_num_seqs}" if strategy.max_num_seqs is not None else ""
        log_path = os.path.join(self.results_dir, f"vllm_{strategy.name}{log_suffix}.log")
        os.makedirs(self.results_dir, exist_ok=True)

        logger.info(f"Launching vLLM [{strategy.name}] on port {port}")
        logger.info(f"Python:  {sys.executable}")
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"Log:     {log_path}")

        env = os.environ.copy()
        # Pin this vLLM instance to its assigned physical GPU(s). vLLM otherwise
        # defaults to GPU 0 regardless of --gpu-indices (which only feeds logging
        # and the TP-count guard), so without this two parallel jobs both land on
        # GPU 0 and OOM. CUDA_VISIBLE_DEVICES also renumbers devices to 0..N-1
        # inside the process, so TP still sees a contiguous device list.
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in self.gpu_indices)

        # Triton flash attention uses MMA kernels that require Ampere+ (compute 8.0+).
        # V100 is Volta (compute 7.0) — disable to fall back to xformers/standard attention.
        env["VLLM_USE_TRITON_FLASH_ATTN"] = "0"

        # FlashInfer's top-k/top-p sampler JIT-compiles a CUDA kernel on first use via
        # nvcc + ninja. This host has no system CUDA toolkit (no /usr/local/cuda, no
        # nvcc on PATH), so that JIT compile fails and crashes the engine process after
        # vLLM otherwise starts up successfully. Force the PyTorch-native sampler instead.
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

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
        self._check_quantization_support(strategy)
        self._check_chunked_prefill_support(strategy)
        _log_system_state(f"before-start:{strategy.name}", port)
        # Evict any stale port-holder or GPU-resident engine worker from a prior run
        self._kill_port_holder(port)
        self._kill_gpu_workers(port)
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
        self._kill_gpu_workers(port)

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

    def _kill_gpu_workers(self, port: int) -> None:
        """Instance-callable wrapper around the module-level cleanup function,
        scoped to this launcher's GPU(s) and the port being (re)started so a
        sibling parallel job on another GPU/port is never swept away."""
        kill_stale_gpu_workers(gpu_indices=self.gpu_indices, ports=[port])

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


def _proc_matches_port(pid: int, port_set: set[int]) -> bool:
    """True if process `pid`'s command line contains `--port <p>` for some p in
    `port_set`. Used to scope the frontend sweep to this job's own server(s)."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            argv = f.read().split(b"\x00")
    except OSError:
        return False
    for i, tok in enumerate(argv):
        if tok == b"--port" and i + 1 < len(argv):
            try:
                if int(argv[i + 1]) in port_set:
                    return True
            except ValueError:
                pass
    return False


def _gpu_uuids_for_indices(gpu_indices: list[int]) -> set[str]:
    """Map physical GPU indices → their UUIDs (as reported by nvidia-smi).

    Needed because nvidia-smi's compute-app listing identifies a process's GPU by
    UUID, not index. Returns an empty set if the mapping can't be built (caller
    treats that as "can't scope safely" and skips the GPU sweep).
    """
    uuids: set[str] = set()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        for line in result.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            if idx in gpu_indices:
                uuids.add(parts[1])
    except Exception as e:
        logger.debug(f"_gpu_uuids_for_indices: {e}")
    return uuids


def kill_stale_gpu_workers(
    gpu_indices: Optional[list[int]] = None,
    ports: Optional[list[int]] = None,
) -> None:
    """SIGKILL stale vLLM processes: both server frontends and GPU engine workers.

    Two sources of stale processes:
    1. vLLM API server frontends (vllm.entrypoints.openai.api_server) — these are
       the parent processes that may survive from a previous TUI session. Even if
       their GPU worker died, the frontend can respawn a new worker and re-occupy
       the GPU when the next run starts.
    2. GPU engine workers (pt_main_thread) — spawned via multiprocessing, escape
       the parent process group, and hold all GPU memory indefinitely after a crash.

    Scoping (for running two jobs in parallel, one model per GPU): when `ports`
    and/or `gpu_indices` are given, only processes belonging to THIS job are
    swept — frontends are matched by their `--port`, GPU workers by the physical
    GPU they occupy — so a sibling job's live server on another port/GPU is never
    mistaken for a stale leftover and killed. With both None the sweep is
    machine-global (single-job / legacy behaviour).
    """
    killed: list[str] = []
    port_set = set(ports) if ports else None

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
            # When port-scoped, only kill the frontend bound to one of our ports;
            # a sibling job's server (different --port) must survive.
            if port_set is not None and not _proc_matches_port(pid, port_set):
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(f"vLLM server PID {pid}")
            except ProcessLookupError:
                pass
    except Exception as e:
        logger.debug(f"kill_stale_gpu_workers (server sweep): {e}")

    # ── Kill GPU-resident engine workers ─────────────────────────────────────
    # When gpu-scoped, restrict to workers resident on THIS job's physical GPU(s)
    # so a sibling job's workers on another GPU are left alone. If we can't build
    # the index→UUID map, skip the GPU sweep entirely rather than risk killing the
    # sibling (the port-scoped frontend sweep above already covers our own leftovers).
    target_uuids: Optional[set[str]] = None
    if gpu_indices is not None:
        target_uuids = _gpu_uuids_for_indices(gpu_indices)
        if not target_uuids:
            logger.debug("kill_stale_gpu_workers: no UUID map; skipping GPU sweep")
            for entry in killed:
                logger.info(f"Killed stale {entry}")
            return
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        for line in result.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 3:
                continue
            uuid_str, pid_str, mem_str = parts
            try:
                pid = int(pid_str)
                mem_mib = int(mem_str)
            except ValueError:
                continue
            if target_uuids is not None and uuid_str not in target_uuids:
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
