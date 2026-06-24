"""
GPU resource monitor.

Runs as a background asyncio task during benchmark execution.
Samples GPU utilization, memory, power, and temperature at regular intervals.
Results are stored per-strategy for post-hoc analysis.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

SAMPLE_INTERVAL_S = 1.0     # 1 Hz sampling


@dataclass
class GPUSample:
    timestamp: float
    gpu_index: int
    util_compute_pct: float     # SM utilization %
    util_memory_pct: float      # memory bus utilization %
    mem_used_mib: float
    mem_total_mib: float
    power_w: float
    temperature_c: float

    @property
    def mem_used_pct(self) -> float:
        return (self.mem_used_mib / self.mem_total_mib) * 100.0 if self.mem_total_mib > 0 else 0.0


@dataclass
class GPUMonitorResult:
    strategy_name: str
    samples: list[GPUSample] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        utils = [s.util_compute_pct for s in self.samples]
        mems  = [s.mem_used_pct    for s in self.samples]
        powers = [s.power_w        for s in self.samples]

        def p(lst, pct): 
            lst_s = sorted(lst)
            idx = int(len(lst_s) * pct / 100)
            return lst_s[min(idx, len(lst_s) - 1)]

        return {
            "gpu_util_mean":   sum(utils) / len(utils),
            "gpu_util_p95":    p(utils, 95),
            "mem_util_mean":   sum(mems) / len(mems),
            "mem_util_p95":    p(mems, 95),
            "power_mean_w":    sum(powers) / len(powers),
            "power_peak_w":    max(powers),
            "sample_count":    len(self.samples),
        }


class GPUMonitor:
    """
    Background GPU sampler. Uses pynvml if available; falls back to nvidia-smi subprocess.
    """

    def __init__(self, gpu_indices: Optional[list[int]] = None):
        self.gpu_indices = gpu_indices or [0]
        self._nvml_available = False
        self._handles = {}
        self._task: Optional[asyncio.Task] = None
        self._result: Optional[GPUMonitorResult] = None
        self._stop_event = asyncio.Event()
        self._init_nvml()

    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            for idx in self.gpu_indices:
                self._handles[idx] = pynvml.nvmlDeviceGetHandleByIndex(idx)
                name = pynvml.nvmlDeviceGetName(self._handles[idx])
                mem  = pynvml.nvmlDeviceGetMemoryInfo(self._handles[idx])
                logger.info(f"GPU {idx}: {name} — {mem.total / (1024**3):.1f} GiB total, {mem.free / (1024**3):.1f} GiB free")
            self._nvml_available = True
            logger.info("pynvml initialized successfully.")
        except Exception as e:
            logger.warning(f"pynvml unavailable ({e}); falling back to nvidia-smi.")

    def _sample_nvml(self) -> list[GPUSample]:
        import pynvml
        samples = []
        t = time.time()
        for idx, handle in self._handles.items():
            util  = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem   = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0   # mW → W
            temp  = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            samples.append(GPUSample(
                timestamp=t,
                gpu_index=idx,
                util_compute_pct=float(util.gpu),
                util_memory_pct=float(util.memory),
                mem_used_mib=mem.used / (1024 ** 2),
                mem_total_mib=mem.total / (1024 ** 2),
                power_w=power,
                temperature_c=float(temp),
            ))
        return samples

    async def _sample_smi(self) -> list[GPUSample]:
        """Fallback: parse nvidia-smi dmon output."""
        import asyncio.subprocess as asp
        t = time.time()
        samples = []
        try:
            idx_str = ",".join(str(i) for i in self.gpu_indices)
            proc = await asp.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,"
                "memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
                f"--id={idx_str}",
                stdout=asp.PIPE,
                stderr=asp.DEVNULL,
            )
            stdout, _ = await proc.communicate()
            for line in stdout.decode().strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 7:
                    continue
                try:
                    samples.append(GPUSample(
                        timestamp=t,
                        gpu_index=int(parts[0]),
                        util_compute_pct=float(parts[1]),
                        util_memory_pct=float(parts[2]),
                        mem_used_mib=float(parts[3]),
                        mem_total_mib=float(parts[4]),
                        power_w=float(parts[5]),
                        temperature_c=float(parts[6]),
                    ))
                except ValueError:
                    pass
        except FileNotFoundError:
            logger.warning("nvidia-smi not found. GPU monitoring disabled.")
        return samples

    async def _monitor_loop(self, result: GPUMonitorResult):
        while not self._stop_event.is_set():
            try:
                if self._nvml_available:
                    samples = self._sample_nvml()
                else:
                    samples = await self._sample_smi()
                result.samples.extend(samples)
            except Exception as e:
                logger.debug(f"GPU sample error: {e}")

            try:
                await asyncio.wait_for(
                    asyncio.shield(self._stop_event.wait()),
                    timeout=SAMPLE_INTERVAL_S,
                )
            except asyncio.TimeoutError:
                pass

    def start(self, strategy_name: str) -> GPUMonitorResult:
        """Start background monitoring. Returns result container."""
        self._result = GPUMonitorResult(strategy_name=strategy_name)
        self._stop_event.clear()
        self._task = asyncio.ensure_future(self._monitor_loop(self._result))
        logger.info(f"GPU monitor started for [{strategy_name}] (backend: {'pynvml' if self._nvml_available else 'nvidia-smi'})")
        return self._result

    async def stop(self) -> GPUMonitorResult:
        """Stop monitoring and return collected samples."""
        self._stop_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except asyncio.TimeoutError:
                self._task.cancel()
        n = len(self._result.samples)
        summary = self._result.summary()
        logger.info(f"GPU monitor stopped. {n} samples collected.")
        if summary:
            logger.info(
                f"GPU summary — util: {summary['gpu_util_mean']:.1f}% mean / {summary['gpu_util_p95']:.1f}% p95 | "
                f"mem: {summary['mem_util_mean']:.1f}% mean | power: {summary['power_mean_w']:.0f}W mean / {summary['power_peak_w']:.0f}W peak"
            )
        return self._result
