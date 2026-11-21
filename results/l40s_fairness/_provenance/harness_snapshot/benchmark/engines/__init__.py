"""
Engine registry: pluggable serving backends (vLLM, SGLang, ...).
"""

import logging
import os
import signal
import subprocess

from benchmark.engines.base import Engine, EngineInstance, ServerMetrics
from benchmark.engines.vllm_engine import VLLMEngine

logger = logging.getLogger(__name__)

ENGINES: dict[str, type[Engine]] = {
    "vllm": VLLMEngine,
}

try:
    from benchmark.engines.sglang_engine import SGLangEngine
    ENGINES["sglang"] = SGLangEngine
except ImportError:
    pass

try:
    from benchmark.engines.tensorrtllm_engine import TensorRTLLMEngine
    ENGINES["tensorrtllm"] = TensorRTLLMEngine
except ImportError:
    pass

try:
    from benchmark.engines.tgi_engine import TGIEngine
    ENGINES["tgi"] = TGIEngine
except ImportError:
    pass


def get_engine(name: str) -> type[Engine]:
    if name not in ENGINES:
        raise ValueError(f"Unknown engine '{name}'. Available: {list(ENGINES.keys())}")
    return ENGINES[name]


def kill_stale_gpu_workers() -> None:
    """SIGKILL any stale serving-engine processes: both server frontends and
    GPU engine workers, across every registered engine.

    Two sources of stale processes:
    1. Server frontends (e.g. vllm.entrypoints.openai.api_server,
       sglang.launch_server) — parent processes that may survive from a
       previous TUI session. Even if their GPU worker died, the frontend can
       respawn a new worker and re-occupy the GPU when the next run starts.
    2. GPU engine workers (e.g. pt_main_thread) — spawned via multiprocessing,
       escape the parent process group, and hold all GPU memory indefinitely
       after a crash.
    """
    killed: list[str] = []
    own_pid = os.getpid()

    # ── Kill server frontends for every registered engine ────────────────
    patterns: list[str] = []
    for engine_cls in ENGINES.values():
        patterns += engine_cls.process_match_patterns()

    for pattern in patterns:
        try:
            result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
            for line in result.stdout.strip().splitlines():
                try:
                    pid = int(line.strip())
                except ValueError:
                    continue
                if pid == own_pid:
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed.append(f"{pattern} PID {pid}")
                except ProcessLookupError:
                    pass
        except Exception as e:
            logger.debug(f"kill_stale_gpu_workers (pattern={pattern}): {e}")

    # ── Kill GPU-resident engine workers (engine-agnostic sweep) ─────────
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
