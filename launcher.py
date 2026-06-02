"""
vLLM instance launcher.

Manages subprocess lifecycle for vLLM OpenAI-compatible API servers.
Each strategy requires its own vLLM instance with a dedicated port.
Supports sequential launch (one strategy at a time) to avoid GPU contention.
"""

import asyncio
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

from config.strategies import Strategy

logger = logging.getLogger(__name__)

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
    ):
        self.model = model
        self.results_dir = results_dir
        self.extra_vllm_args = extra_vllm_args or []
        self._current_instance: Optional[VLLMInstance] = None

    def _build_command(self, strategy: Strategy, port: int) -> list[str]:
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
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
        logger.debug(f"Command: {' '.join(cmd)}")

        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,   # process group for clean teardown
            )
        return process

    async def _wait_for_health(self, instance: VLLMInstance) -> bool:
        """Poll /health until vLLM is ready or timeout."""
        deadline = time.time() + VLLM_STARTUP_TIMEOUT_S
        async with aiohttp.ClientSession() as session:
            while time.time() < deadline:
                try:
                    async with session.get(
                        instance.health_url, timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            logger.info(
                                f"vLLM [{instance.strategy.name}] healthy "
                                f"(startup: {instance.startup_time_s:.1f}s)"
                            )
                            return True
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
                await asyncio.sleep(HEALTH_POLL_INTERVAL_S)
        logger.error(
            f"vLLM [{instance.strategy.name}] failed to start within "
            f"{VLLM_STARTUP_TIMEOUT_S}s"
        )
        return False

    async def start(self, strategy: Strategy, port: int = BASE_PORT) -> VLLMInstance:
        """Launch vLLM for the given strategy and wait until healthy."""
        t0 = time.time()
        process = self._launch_process(strategy, port)
        instance = VLLMInstance(strategy=strategy, model=self.model, port=port, process=process)

        # Poll for readiness
        healthy = await self._wait_for_health(instance)
        instance.startup_time_s = time.time() - t0

        if not healthy:
            self.stop(instance)
            raise RuntimeError(f"vLLM instance [{strategy.name}] failed to start.")

        self._current_instance = instance
        return instance

    def stop(self, instance: Optional[VLLMInstance] = None) -> None:
        """Terminate the vLLM process and release GPU memory."""
        target = instance or self._current_instance
        if target is None or target.process is None:
            return

        logger.info(f"Stopping vLLM [{target.strategy.name}]")
        try:
            os.killpg(os.getpgid(target.process.pid), signal.SIGTERM)
            target.process.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(target.process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

        target.process = None
        self._current_instance = None

        # Brief pause to ensure GPU memory is fully released
        time.sleep(5)
        logger.info(f"vLLM [{target.strategy.name}] stopped.")

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
