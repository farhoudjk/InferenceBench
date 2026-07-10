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
    log_path: str = ""
    # KV-cache capacity in tokens (how much decode state the GPU can hold at once).
    # Used by the load guardrail to warn when offered concurrency exceeds what the
    # card can actually run, which otherwise shows up as huge TTFT from queueing.
    kv_token_capacity: int = 0
    kv_capacity_source: str = "unavailable"

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
        # Incremented per server launch so each config gets its own log file —
        # otherwise configs sharing a strategy+seqs suffix reuse one log and
        # overwrite the previous one, destroying the crash trace of any server
        # that died mid-run.
        self._launch_seq = 0

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
        # Also prefix a per-launch sequence number so configs sharing the same
        # strategy+seqs (different workload/rate) each keep their own log and a
        # mid-run crash trace is never overwritten by the next config.
        self._launch_seq += 1
        log_suffix = f"_seqs{strategy.max_num_seqs}" if strategy.max_num_seqs is not None else ""
        log_path = os.path.join(
            self.results_dir,
            f"vllm_{strategy.name}{log_suffix}_{self._launch_seq:02d}.log",
        )
        os.makedirs(self.results_dir, exist_ok=True)

        self._last_log_path = log_path
        logger.info(f"Launching vLLM [{strategy.name}] on port {port}")
        logger.info(f"Python:  {sys.executable}")
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"Log:     {log_path}")

        env = os.environ.copy()
        # Triton flash attention uses MMA kernels that require Ampere+ (compute 8.0+).
        # V100 is Volta (compute 7.0) — disable to fall back to xformers/standard attention.
        env["VLLM_USE_TRITON_FLASH_ATTN"] = "0"

        # Pin vLLM to the GPU(s) chosen in the TUI/CLI. Without this, CUDA hands the
        # subprocess physical GPU 0 by default regardless of --gpu-indices, so a busy
        # GPU 0 causes CUDA OOM even when the chosen GPU is free — and a run can never
        # actually use GPU 1. vLLM sees the exposed GPUs renumbered from 0, so its
        # tensor-parallel workers map onto exactly the selected devices. Set only on
        # the subprocess env (not os.environ), so the parent's diagnostics and the
        # NVML monitor keep physical indexing.
        if self.gpu_indices:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in self.gpu_indices)

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

        # Record how much KV cache the card actually has, so the runner's load
        # guardrail can warn when offered concurrency exceeds real capacity.
        instance.log_path = getattr(self, "_last_log_path", "")
        cap, source = estimate_kv_capacity(
            self.model, strategy, self.gpu_indices, log_path=instance.log_path
        )
        instance.kv_token_capacity = cap
        instance.kv_capacity_source = source
        if cap:
            logger.info(
                f"KV cache capacity: ~{cap:,} tokens "
                f"(~{max(1, cap // max(1, strategy.max_model_len))} sequences at "
                f"max_model_len={strategy.max_model_len}) — {source}"
            )

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
        """Instance-callable wrapper around the module-level cleanup function.

        Scoped to this launcher's GPU(s) and the given port so it evicts only
        this run's stale workers — never a concurrent run's server on another GPU.
        """
        kill_stale_gpu_workers(gpu_indices=self.gpu_indices, port=port)

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


# vLLM's default KV-cache block size (tokens per block). The benchmark never
# overrides --block-size, so this holds.
_VLLM_BLOCK_SIZE = 16


def _parse_gpu_blocks_from_log(log_path: str) -> Optional[int]:
    """Extract vLLM's profiled '# GPU blocks: N' from its startup log.

    This is ground truth: vLLM measures free memory after loading weights and
    profiling peak activation, then reports how many KV blocks fit. Returns the
    last such count (there may be one per engine), or None if not present.
    """
    if not log_path:
        return None
    try:
        with open(log_path, "r", errors="ignore") as f:
            text = f.read()
    except OSError:
        return None
    matches = re.findall(r"GPU blocks:\s*(\d+)", text)
    return int(matches[-1]) if matches else None


def _estimate_param_bytes(cfg, dtype_bytes: float = 2.0) -> int:
    """Rough transformer weight footprint from an HF config (norms/biases ignored)."""
    V = getattr(cfg, "vocab_size", 32000)
    H = cfg.hidden_size
    L = cfg.num_hidden_layers
    I = getattr(cfg, "intermediate_size", 4 * H)
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", None) or n_heads
    hd = getattr(cfg, "head_dim", None) or (H // n_heads)
    embed = V * H
    attn = 2 * H * n_heads * hd + 2 * H * n_kv * hd   # (q+o) + (k+v), GQA-aware
    mlp = 3 * H * I                                    # gate + up + down
    params = embed + L * (attn + mlp)
    if not getattr(cfg, "tie_word_embeddings", True):
        params += V * H
    return int(params * dtype_bytes)


def _kv_bytes_per_token(cfg) -> int:
    """KV-cache bytes for one token across all layers (fp16/bf16 KV = 2 bytes)."""
    H = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", None) or n_heads
    hd = getattr(cfg, "head_dim", None) or (H // n_heads)
    L = cfg.num_hidden_layers
    return 2 * L * n_kv * hd * 2   # 2 (K+V) × layers × kv_heads × head_dim × 2 bytes


def estimate_kv_capacity(model, strategy, gpu_indices, log_path=None):
    """Return (kv_token_capacity, source). Prefer vLLM's profiled GPU-block count;
    fall back to a model-config + GPU-memory estimate if the log line is absent."""
    blocks = _parse_gpu_blocks_from_log(log_path)
    if blocks:
        return blocks * _VLLM_BLOCK_SIZE, f"vLLM profiled ({blocks} GPU blocks × {_VLLM_BLOCK_SIZE})"
    try:
        from transformers import AutoConfig
        import torch
        cfg = AutoConfig.from_pretrained(model)
        # weight dtype: fp8 ≈ 1 B/param, AWQ/GPTQ 4-bit ≈ 0.5, else fp16 = 2.
        q = (strategy.quantization or "").lower()
        wbytes = 1.0 if "fp8" in q else (0.5 if q in ("awq", "gptq") else 2.0)
        weight_bytes = _estimate_param_bytes(cfg, dtype_bytes=wbytes)
        idx = gpu_indices[0] if gpu_indices else 0
        total_mem = torch.cuda.get_device_properties(idx).total_memory
        tp = max(1, strategy.tensor_parallel_size)
        util = strategy.gpu_memory_utilization or 0.9
        overhead = int(1.5 * 1024 ** 3)   # activations + CUDA graphs + fragmentation
        kv_budget = max(0, int(total_mem * tp * util) - weight_bytes - overhead)
        cap = kv_budget // max(1, _kv_bytes_per_token(cfg))
        return int(cap), "estimated (config + GPU mem)"
    except Exception as e:
        logger.debug(f"KV capacity estimate failed: {e}")
        return 0, "unavailable"


def _proc_cmdline_has_port(pid: int, port: int) -> bool:
    """True if process `pid` was launched with `--port <port>` (exact match).

    Reads /proc/<pid>/cmdline (NUL-separated argv) so we can tell one run's vLLM
    frontend from another's by the port it serves. Returns False if unreadable.
    """
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            argv = f.read().split(b"\x00")
    except OSError:
        return False
    target = str(port).encode()
    for i, tok in enumerate(argv):
        if tok == b"--port=" + target:
            return True
        if tok == b"--port" and i + 1 < len(argv) and argv[i + 1] == target:
            return True
    return False


def kill_stale_gpu_workers(
    gpu_indices: Optional[list[int]] = None,
    port: Optional[int] = None,
) -> None:
    """SIGKILL stale vLLM processes: both server frontends and GPU engine workers.

    Two sources of stale processes:
    1. vLLM API server frontends (vllm.entrypoints.openai.api_server) — these are
       the parent processes that may survive from a previous TUI session. Even if
       their GPU worker died, the frontend can respawn a new worker and re-occupy
       the GPU when the next run starts.
    2. GPU engine workers (pt_main_thread) — spawned via multiprocessing, escape
       the parent process group, and hold all GPU memory indefinitely after a crash.

    Scoping (both optional, default = machine-wide sweep):
    - port: only kill frontends bound to this --port. Without it, a second run
      starting on another GPU would SIGKILL the first run's still-serving
      frontend (they all match the api_server pattern).
    - gpu_indices: only kill GPU workers resident on these physical GPUs. Without
      it, the nvidia-smi sweep spans every GPU and kills a concurrent run's
      worker. Passing this run's GPUs makes two runs on separate GPUs coexist.
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
            # When scoped to a port, leave frontends bound to any other port
            # alone — they belong to a different (possibly concurrent) run.
            if port is not None and not _proc_cmdline_has_port(pid, port):
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
        query_cmd = ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"]
        # Restrict the sweep to this run's physical GPUs so a concurrent run's
        # worker on another GPU is never touched. nvidia-smi -i uses physical
        # indexing, matching gpu_indices.
        if gpu_indices:
            query_cmd += ["-i", ",".join(str(i) for i in gpu_indices)]
        result = subprocess.run(query_cmd, capture_output=True, text=True)
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
