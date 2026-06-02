"""
Async benchmark runner.

Replays a request trace against a live vLLM instance.
Measures TTFT (Time to First Token) and E2E latency via SSE streaming.
Respects arrival times to simulate realistic load patterns.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

from benchmark.launcher import VLLMInstance, get_vllm_metrics
from benchmark.gpu_monitor import GPUMonitor, GPUMonitorResult
from config.strategies import Strategy
from config.workloads import RequestSpec, WorkloadConfig, build_trace

logger = logging.getLogger(__name__)

# ── Per-request result ────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    request_id: str
    strategy_name: str
    workload_name: str
    task_type: str
    prompt_token_len: int
    target_output_tokens: int
    arrival_time: float

    # measured
    send_time: float = 0.0          # wall-clock when request was dispatched
    ttft_s: float = 0.0             # time to first token
    e2e_s: float = 0.0              # total request latency
    output_tokens: int = 0          # actual tokens generated
    tokens_per_second: float = 0.0  # generation throughput
    status: str = "pending"         # "ok" | "error" | "timeout"
    error: str = ""

    # vLLM server metrics at request completion
    kv_cache_usage_pct: float = 0.0
    num_waiting_seqs: int = 0
    num_running_seqs: int = 0
    num_preemptions: int = 0


# ── Benchmark result container ────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    strategy: Strategy
    workload_name: str
    requests: list[RequestResult] = field(default_factory=list)
    gpu_result: Optional[GPUMonitorResult] = None
    wall_time_s: float = 0.0
    vllm_startup_time_s: float = 0.0

    def successful(self) -> list[RequestResult]:
        return [r for r in self.requests if r.status == "ok"]

    def summary(self) -> dict:
        ok = self.successful()
        if not ok:
            return {"strategy": self.strategy.name, "n_ok": 0, "n_total": len(self.requests)}

        def pct(lst, p):
            s = sorted(lst)
            idx = max(0, min(int(len(s) * p / 100), len(s) - 1))
            return s[idx]

        ttfts = [r.ttft_s * 1000 for r in ok]      # ms
        e2es  = [r.e2e_s  * 1000 for r in ok]       # ms
        tpss  = [r.tokens_per_second for r in ok]

        return {
            "strategy":       self.strategy.name,
            "workload":       self.workload_name,
            "n_ok":           len(ok),
            "n_error":        len(self.requests) - len(ok),
            "ttft_p50_ms":    pct(ttfts, 50),
            "ttft_p95_ms":    pct(ttfts, 95),
            "ttft_p99_ms":    pct(ttfts, 99),
            "e2e_p50_ms":     pct(e2es, 50),
            "e2e_p95_ms":     pct(e2es, 95),
            "e2e_p99_ms":     pct(e2es, 99),
            "tps_mean":       sum(tpss) / len(tpss),
            "tps_p50":        pct(tpss, 50),
            "wall_time_s":    self.wall_time_s,
            "throughput_rps": len(ok) / self.wall_time_s if self.wall_time_s > 0 else 0,
        }


# ── SSE streaming request sender ─────────────────────────────────────────────

async def _send_request(
    session: aiohttp.ClientSession,
    instance: VLLMInstance,
    req: RequestSpec,
    strategy_name: str,
    workload_name: str,
    timeout_s: float = 300.0,
) -> RequestResult:
    result = RequestResult(
        request_id=req.request_id,
        strategy_name=strategy_name,
        workload_name=workload_name,
        task_type=req.task_type.value,
        prompt_token_len=req.prompt_token_len,
        target_output_tokens=req.target_output_tokens,
        arrival_time=req.arrival_time,
    )

    payload = {
        "model": instance.model,
        "prompt": req.prompt,
        "max_tokens": req.target_output_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_send = time.perf_counter()
    result.send_time = time.time()
    first_token_received = False
    total_completion_tokens = 0

    try:
        async with session.post(
            instance.completions_url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            if resp.status != 200:
                result.status = "error"
                result.error = f"HTTP {resp.status}"
                return result

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue

                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # TTFT: first chunk with a non-empty token
                if not first_token_received:
                    choices = chunk.get("choices", [])
                    if choices and choices[0].get("text", ""):
                        result.ttft_s = time.perf_counter() - t_send
                        first_token_received = True

                # token count from usage field (present in last chunk when include_usage=True)
                usage = chunk.get("usage")
                if usage:
                    total_completion_tokens = usage.get("completion_tokens", 0)

        t_end = time.perf_counter()
        result.e2e_s = t_end - t_send
        result.output_tokens = total_completion_tokens

        if result.e2e_s > 0 and total_completion_tokens > 0:
            decode_time = result.e2e_s - result.ttft_s
            result.tokens_per_second = (
                total_completion_tokens / decode_time if decode_time > 0 else 0.0
            )

        result.status = "ok" if first_token_received else "error"
        if not first_token_received:
            result.error = "no tokens received"

    except asyncio.TimeoutError:
        result.status = "timeout"
        result.error = f"timeout after {timeout_s}s"
    except aiohttp.ClientError as e:
        result.status = "error"
        result.error = str(e)

    return result


# ── Trace replayer ────────────────────────────────────────────────────────────

async def run_benchmark(
    instance: VLLMInstance,
    workload_config: WorkloadConfig,
    gpu_monitor: Optional[GPUMonitor] = None,
    request_timeout_s: float = 300.0,
    max_concurrent: int = 64,
    metrics_poll_interval_s: float = 5.0,
    seed: int = 42,
) -> BenchmarkResult:
    """
    Replay workload trace against vLLM instance.
    Respects arrival times (Poisson / bursty / uniform).
    Returns BenchmarkResult with per-request metrics.
    """
    strategy = instance.strategy
    trace = build_trace(workload_config, seed=seed)

    result = BenchmarkResult(
        strategy=strategy,
        workload_name=workload_config.name,
        vllm_startup_time_s=instance.startup_time_s,
    )

    gpu_result = gpu_monitor.start(strategy.name) if gpu_monitor else None

    semaphore = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(limit=max_concurrent + 10)
    completed_results: list[RequestResult] = []

    # Background task: poll vLLM server metrics periodically
    server_metrics_cache: dict = {}

    async def _poll_server_metrics():
        while True:
            m = await get_vllm_metrics(instance)
            server_metrics_cache.update(m)
            await asyncio.sleep(metrics_poll_interval_s)

    async def _dispatch(session: aiohttp.ClientSession, req: RequestSpec):
        async with semaphore:
            r = await _send_request(
                session, instance, req,
                strategy_name=strategy.name,
                workload_name=workload_config.name,
                timeout_s=request_timeout_s,
            )
            # attach latest server metrics snapshot
            r.kv_cache_usage_pct = server_metrics_cache.get("vllm:gpu_cache_usage_perc", 0.0)
            r.num_waiting_seqs   = int(server_metrics_cache.get("vllm:num_waiting_seqs", 0))
            r.num_running_seqs   = int(server_metrics_cache.get("vllm:num_running_seqs", 0))
            r.num_preemptions    = int(server_metrics_cache.get("vllm:num_preemptions_total", 0))
            completed_results.append(r)

            status_icon = "✓" if r.status == "ok" else "✗"
            logger.debug(
                f"{status_icon} [{req.request_id}] "
                f"TTFT={r.ttft_s*1000:.0f}ms E2E={r.e2e_s*1000:.0f}ms "
                f"tokens={r.output_tokens}"
            )

    trace_start = time.perf_counter()
    t_wall_start = time.time()

    async with aiohttp.ClientSession(connector=connector) as session:
        metrics_task = asyncio.create_task(_poll_server_metrics())
        tasks = []

        for req in trace:
            # Wait until this request's scheduled arrival time
            elapsed = time.perf_counter() - trace_start
            delay = req.arrival_time - elapsed
            if delay > 0:
                await asyncio.sleep(delay)

            tasks.append(asyncio.create_task(_dispatch(session, req)))

        # Wait for all in-flight requests to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        metrics_task.cancel()

    result.wall_time_s = time.time() - t_wall_start
    result.requests = completed_results

    if gpu_monitor:
        result.gpu_result = await gpu_monitor.stop()

    n_ok = len(result.successful())
    logger.info(
        f"Benchmark [{strategy.name}] x [{workload_config.name}] complete: "
        f"{n_ok}/{len(trace)} ok, wall={result.wall_time_s:.1f}s"
    )
    if n_ok > 0:
        s = result.summary()
        logger.info(
            f"  TTFT P95={s['ttft_p95_ms']:.0f}ms  "
            f"E2E P95={s['e2e_p95_ms']:.0f}ms  "
            f"TPS={s['tps_mean']:.1f}"
        )

    return result
