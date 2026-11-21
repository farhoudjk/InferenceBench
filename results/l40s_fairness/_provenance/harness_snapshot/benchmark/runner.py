"""
Async benchmark runner.

Replays a request trace against a live vLLM instance.
Measures TTFT (Time to First Token) and E2E latency via SSE streaming.
Respects arrival times to simulate realistic load patterns.
"""

import asyncio
import functools
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

from benchmark.engines.base import Engine, EngineInstance, ServerMetrics
from benchmark.gpu_monitor import GPUMonitor, GPUMonitorResult
from config.strategies import Strategy
from config.workloads import (
    ArrivalPattern, PromptSource, RequestSpec, WorkloadConfig,
    build_conversation_trace, build_trace, _count_tokens,
)

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=8)
def _get_chat_template_fn(model_path: str):
    """Returns a callable(prompt: str) -> str that wraps a plain-text prompt
    in model_path's own chat template, or None if the model's tokenizer has
    no chat template configured.

    Every workload here builds plain instruction-style text (see
    config/workloads.py's _make_prompt and the real-dataset loaders) and this
    runner sends it straight to /v1/completions with no chat wrapper. Llama-
    3.1-Instruct and Mistral-Nemo-Instruct tolerate that reasonably (they
    still produce real completions, just not "chat-shaped" ones). Qwen1.5-
    Chat does not: confirmed live via a direct /v1/completions probe that it
    deterministically (temperature=0) emits its <|im_end|> stop token as the
    very first generated token when given unwrapped text, producing a 0-
    token completion on every request. Applying the tokenizer's own
    apply_chat_template() before sending fixes this without hardcoding any
    per-model-name logic — it's a no-op for tokenizers with no chat_template.
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        logger.warning(f"Could not load tokenizer for {model_path!r} to check for a chat template: {e}")
        return None
    if getattr(tok, "chat_template", None) is None:
        return None

    def apply(prompt: str) -> str:
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return apply


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
    response_text: str = ""         # accumulated generated text (used to build
                                     # the next turn's growing prompt in
                                     # multi-turn conversations; empty/unused
                                     # for independent single-turn requests)

    # multi-turn conversation fields (None for independent requests)
    conversation_id: Optional[str] = None
    turn_index: Optional[int] = None
    num_turns: Optional[int] = None

    # server-side scheduler metrics at request completion (None if the
    # engine doesn't expose an equivalent — distinct from "measured zero")
    kv_cache_usage_pct: Optional[float] = None
    num_waiting_seqs: Optional[int] = None
    num_running_seqs: Optional[int] = None
    num_preemptions: Optional[int] = None
    num_preemptions_delta: Optional[int] = None  # preemptions during this request's lifetime

    # speculative decoding acceptance, sampled from server metrics at completion
    spec_draft_acceptance_rate: Optional[float] = None
    spec_decode_efficiency: Optional[float] = None

    # prefix/KV-cache reuse rate, sampled from server metrics at completion
    # (None if the engine doesn't expose an equivalent — see ServerMetrics)
    prefix_cache_hit_rate: Optional[float] = None

    # queueing delay: time from dispatch to first byte of any HTTP response
    # (separates scheduler queueing from model compute time, which starts at TTFT)
    ttfb_s: float = 0.0

    # inter-token latency (decode-phase stalls), computed from per-chunk timestamps
    itl_mean_ms: float = 0.0
    itl_p50_ms: float = 0.0
    itl_p95_ms: float = 0.0
    itl_max_ms: float = 0.0

    # derived ratios
    completion_ratio: float = 0.0   # output_tokens / target_output_tokens


# ── Benchmark result container ────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    strategy: Strategy
    workload_name: str
    requests: list[RequestResult] = field(default_factory=list)
    gpu_result: Optional[GPUMonitorResult] = None
    wall_time_s: float = 0.0
    vllm_startup_time_s: float = 0.0
    engine_name: str = ""          # "vllm" | "sglang" | "tensorrtllm"
    rate_label: str = ""          # e.g. "low_poisson", "high_bursty"
    effective_rps: float = 0.0    # actual rps used (0 for bursty)
    arrival_pattern: str = ""     # "poisson" | "bursty"
    max_concurrent: int = 0       # concurrency cap (semaphore limit) used for this run

    def successful(self) -> list[RequestResult]:
        return [r for r in self.requests if r.status == "ok"]

    def summary(self, ttft_slo_ms: float = 200.0) -> dict:
        ok = self.successful()
        if not ok:
            return {"strategy": self.strategy.name, "n_ok": 0, "n_total": len(self.requests)}

        def pct(lst, p):
            s = sorted(lst)
            idx = max(0, min(int(len(s) * p / 100), len(s) - 1))
            return s[idx]

        ttfts = [r.ttft_s * 1000 for r in ok]      # ms
        ttfbs = [r.ttfb_s * 1000 for r in ok]      # ms
        e2es  = [r.e2e_s  * 1000 for r in ok]       # ms
        tpss  = [r.tokens_per_second for r in ok]
        itls  = [r.itl_mean_ms for r in ok if r.itl_mean_ms > 0]
        completion_ratios = [r.completion_ratio for r in ok if r.completion_ratio > 0]
        accept_rates = [
            r.spec_draft_acceptance_rate for r in ok
            if r.spec_draft_acceptance_rate is not None and r.spec_draft_acceptance_rate > 0
        ]
        preemption_deltas = [r.num_preemptions_delta for r in ok if r.num_preemptions_delta is not None]
        total_preemptions = sum(preemption_deltas) if preemption_deltas else None

        n_slo_ok = sum(1 for t in ttfts if t < ttft_slo_ms)

        result = {
            "strategy":       self.strategy.name,
            "engine":         self.engine_name,
            "workload":       self.workload_name,
            "rate_label":     self.rate_label,
            "effective_rps":  self.effective_rps,
            "arrival_pattern": self.arrival_pattern,
            "max_concurrent": self.max_concurrent,
            "max_num_seqs": self.strategy.max_num_seqs,
            "n_ok":           len(ok),
            "n_error":        len(self.requests) - len(ok),
            "ttft_p50_ms":    pct(ttfts, 50),
            "ttft_p95_ms":    pct(ttfts, 95),
            "ttft_p99_ms":    pct(ttfts, 99),
            "ttfb_p50_ms":    pct(ttfbs, 50),
            "ttfb_p95_ms":    pct(ttfbs, 95),
            # queueing delay: gap between any-byte-received (TTFB) and first real token (TTFT)
            "queue_delay_p50_ms": max(0.0, pct(ttfts, 50) - pct(ttfbs, 50)),
            "queue_delay_p95_ms": max(0.0, pct(ttfts, 95) - pct(ttfbs, 95)),
            "e2e_p50_ms":     pct(e2es, 50),
            "e2e_p95_ms":     pct(e2es, 95),
            "e2e_p99_ms":     pct(e2es, 99),
            "tps_mean":       sum(tpss) / len(tpss),
            "tps_p50":        pct(tpss, 50),
            "wall_time_s":    self.wall_time_s,
            "throughput_rps": len(ok) / self.wall_time_s if self.wall_time_s > 0 else 0,
            f"slo_attainment_ttft_{int(ttft_slo_ms)}ms_pct": 100.0 * n_slo_ok / len(ok),
            "completion_ratio_mean": (sum(completion_ratios) / len(completion_ratios)) if completion_ratios else 0.0,
            "total_preemptions": total_preemptions,
        }

        if itls:
            result["itl_mean_ms"] = sum(itls) / len(itls)
            result["itl_p95_ms"]  = pct(itls, 95)
        if accept_rates:
            result["spec_draft_acceptance_rate_mean"] = sum(accept_rates) / len(accept_rates)

        return result


# ── SSE streaming request sender ─────────────────────────────────────────────

async def _send_request(
    session: aiohttp.ClientSession,
    instance: EngineInstance,
    req: RequestSpec,
    strategy_name: str,
    workload_name: str,
    timeout_s: float = 300.0,
    ttft_timeout_s: float = 120.0,
) -> RequestResult:
    # NOTE: this reads only choices[0].text (SSE chunks) and the final chunk's
    # usage.completion_tokens — both vLLM and SGLang agree on this shape. Known
    # response-field divergences that this function does NOT currently read:
    # vLLM's `stop_reason` vs SGLang's `matched_stop`, and vLLM-only `token_ids`.
    # If a future metric needs one of those, add an
    # `engine.parse_stop_reason(chunk)`-style hook rather than branching here.
    result = RequestResult(
        request_id=req.request_id,
        strategy_name=strategy_name,
        workload_name=workload_name,
        task_type=req.task_type.value,
        prompt_token_len=req.prompt_token_len,
        target_output_tokens=req.target_output_tokens,
        arrival_time=req.arrival_time,
        conversation_id=req.conversation_id,
        turn_index=req.turn_index,
        num_turns=req.num_turns,
    )

    chat_template_fn = _get_chat_template_fn(instance.model)
    prompt = chat_template_fn(req.prompt) if chat_template_fn is not None else req.prompt

    # Clamp max_tokens so prompt + completion never exceeds the server's
    # max_model_len — real-dataset workloads (e.g. OpenMathReasoning CoT
    # traces) can have reference output lengths that alone exceed the
    # context window. A margin absorbs the vLLM tokenizer occasionally
    # counting a few more prompt tokens than our local estimate, plus (when a
    # chat_template_fn applied) the special tokens/role markers it adds on
    # top of req.prompt_token_len, which was counted on the unwrapped text —
    # confirmed live this session: Qwen1.5-Chat's template alone added ~19
    # tokens to a 28-token prompt, well past a margin sized only for
    # tokenizer-count noise.
    context_margin = 16 if chat_template_fn is None else 96
    max_completion_tokens = instance.strategy.max_model_len - req.prompt_token_len - context_margin
    if max_completion_tokens < 1:
        # Prompt alone already fills (or nearly fills) the context window —
        # rare (e.g. ShareGPT's long tail) but no max_tokens value can fix
        # this, so fail fast rather than send a request vLLM will reject.
        result.status = "error"
        result.error = (
            f"prompt_token_len={req.prompt_token_len} leaves no room for output "
            f"under max_model_len={instance.strategy.max_model_len}"
        )
        logger.warning(f"[{req.request_id}] {result.error}")
        return result

    effective_max_tokens = min(req.target_output_tokens, max_completion_tokens)
    if effective_max_tokens < req.target_output_tokens:
        logger.debug(
            f"[{req.request_id}] clamping max_tokens {req.target_output_tokens} -> "
            f"{effective_max_tokens} (prompt_len={req.prompt_token_len}, "
            f"max_model_len={instance.strategy.max_model_len})"
        )

    payload = {
        "model": instance.model,
        "prompt": prompt,
        "max_tokens": effective_max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_send = time.perf_counter()
    result.send_time = time.time()
    first_token_received = False
    first_byte_received = False
    total_completion_tokens = 0
    chunks_received = 0
    token_arrival_times: list[float] = []   # perf_counter() at each non-empty token chunk
    response_text_parts: list[str] = []     # accumulated generated text (see RequestResult.response_text)

    logger.debug(f"[{req.request_id}] sending POST → {instance.completions_url}")

    try:
        async with session.post(
            instance.completions_url,
            json=payload,
            timeout=aiohttp.ClientTimeout(
                total=timeout_s,
                sock_connect=10,   # fail fast if vLLM not accepting connections
                # Must be >= ttft_timeout_s: aiohttp raises its own
                # asyncio.TimeoutError from sock_read before our explicit
                # asyncio.wait_for(..., timeout=remaining) below ever gets a
                # chance to fire, if sock_read is the smaller of the two —
                # silently capping the effective TTFT timeout regardless of
                # what the caller passed. Confirmed live: a hardcoded
                # sock_read=120 here made every request report
                # "ttft_timeout after 120.0s" under real queueing delay even
                # with ttft_timeout_s=300 configured.
                sock_read=max(ttft_timeout_s, timeout_s),
            ),
        ) as resp:
            logger.debug(f"[{req.request_id}] HTTP {resp.status} received")
            if resp.status != 200:
                body = await resp.text()
                result.status = "error"
                result.error = f"HTTP {resp.status}: {body[:200]}"
                logger.warning(f"[{req.request_id}] {result.error}")
                return result

            content_iter = resp.content.__aiter__()
            while True:
                try:
                    if first_token_received:
                        raw_line = await content_iter.__anext__()
                    else:
                        # Bound time-to-first-token specifically, independent of
                        # the overall request timeout — a request stuck queued
                        # behind other work should fail fast rather than tie up
                        # a concurrency slot for the full request timeout.
                        remaining = ttft_timeout_s - (time.perf_counter() - t_send)
                        if remaining <= 0:
                            raise asyncio.TimeoutError("ttft_timeout")
                        raw_line = await asyncio.wait_for(content_iter.__anext__(), timeout=remaining)
                except StopAsyncIteration:
                    break

                if not first_byte_received:
                    # TTFB: first byte of any response, before parsing — separates
                    # scheduler queueing delay from model compute time (TTFT below)
                    result.ttfb_s = time.perf_counter() - t_send
                    first_byte_received = True

                chunks_received += 1
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue

                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    logger.debug(f"[{req.request_id}] [DONE] after {chunks_received} chunks")
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                has_token = bool(choices and choices[0].get("text", ""))

                if has_token:
                    t_now = time.perf_counter()
                    token_arrival_times.append(t_now)
                    response_text_parts.append(choices[0]["text"])
                    if not first_token_received:
                        result.ttft_s = t_now - t_send
                        first_token_received = True
                        logger.debug(f"[{req.request_id}] first token at {result.ttft_s*1000:.0f}ms")

                # token count from usage field (present in last chunk when include_usage=True)
                usage = chunk.get("usage")
                if usage:
                    total_completion_tokens = usage.get("completion_tokens", 0)

        t_end = time.perf_counter()
        result.e2e_s = t_end - t_send
        result.output_tokens = total_completion_tokens
        result.response_text = "".join(response_text_parts)
        if req.target_output_tokens > 0:
            result.completion_ratio = total_completion_tokens / req.target_output_tokens

        if result.e2e_s > 0 and total_completion_tokens > 0:
            decode_time = result.e2e_s - result.ttft_s
            result.tokens_per_second = (
                total_completion_tokens / decode_time if decode_time > 0 else 0.0
            )

        # Inter-token latency: gaps between consecutive token arrivals (decode-phase stalls)
        if len(token_arrival_times) >= 2:
            gaps_ms = [
                (token_arrival_times[i] - token_arrival_times[i - 1]) * 1000
                for i in range(1, len(token_arrival_times))
            ]
            gaps_sorted = sorted(gaps_ms)
            result.itl_mean_ms = sum(gaps_ms) / len(gaps_ms)
            result.itl_p50_ms = gaps_sorted[len(gaps_sorted) // 2]
            result.itl_p95_ms = gaps_sorted[min(int(len(gaps_sorted) * 0.95), len(gaps_sorted) - 1)]
            result.itl_max_ms = max(gaps_ms)

        result.status = "ok" if first_token_received else "error"
        if not first_token_received:
            result.error = f"no tokens received (chunks={chunks_received})"
            logger.warning(f"[{req.request_id}] {result.error}")

    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - t_send
        result.status = "timeout"
        kind = "ttft_timeout" if not first_token_received else "timeout"
        result.error = f"{kind} after {elapsed:.1f}s (first_token={first_token_received}, chunks={chunks_received})"
        logger.warning(f"[{req.request_id}] {result.error}")
    except aiohttp.ClientError as e:
        result.status = "error"
        result.error = f"{type(e).__name__}: {e}"
        logger.warning(f"[{req.request_id}] {result.error}")
    except Exception as e:
        result.status = "error"
        result.error = f"unexpected {type(e).__name__}: {e}"
        logger.error(f"[{req.request_id}] {result.error}", exc_info=True)

    return result


# ── Trace replayer ────────────────────────────────────────────────────────────

async def run_benchmark(
    instance: EngineInstance,
    engine: Engine,
    workload_config: WorkloadConfig,
    gpu_monitor: Optional[GPUMonitor] = None,
    request_timeout_s: float = 300.0,
    ttft_timeout_s: float = 120.0,
    max_concurrent: int = 64,
    metrics_poll_interval_s: float = 5.0,
    seed: int = 42,
    arrival_pattern: Optional[ArrivalPattern] = None,
    target_rps: Optional[float] = None,
    rate_label: str = "",
) -> BenchmarkResult:
    """
    Replay workload trace against a running engine instance.

    arrival_pattern / target_rps override the workload's defaults (set from TUI
    rate selection).  rate_label is a human-readable tag ("low", "med", "high",
    "bursty") stored in the result for grouping in reports.
    """
    strategy = instance.strategy
    is_multiturn = workload_config.prompt_source == PromptSource.SHAREGPT_MULTITURN
    if is_multiturn:
        conversations = build_conversation_trace(
            workload_config, seed=seed,
            arrival_pattern=arrival_pattern,
            target_rps=target_rps,
        )
        trace: list[RequestSpec] = [turn for conv in conversations for turn in conv]
    else:
        trace = build_trace(
            workload_config, seed=seed,
            arrival_pattern=arrival_pattern,
            target_rps=target_rps,
        )

    effective_pattern = arrival_pattern if arrival_pattern is not None else workload_config.default_arrival_pattern
    result = BenchmarkResult(
        strategy=strategy,
        workload_name=workload_config.name,
        vllm_startup_time_s=instance.startup_time_s,
        engine_name=engine.name,
        rate_label=rate_label or workload_config.default_arrival_pattern.value,
        effective_rps=target_rps or workload_config.default_target_rps,
        arrival_pattern=effective_pattern.value,
        max_concurrent=max_concurrent,
    )

    gpu_result = gpu_monitor.start(strategy.name) if gpu_monitor else None

    n_total = len(trace)
    logger.info(f"Starting benchmark: {n_total} requests, max_concurrent={max_concurrent}")

    semaphore = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(
        limit=max_concurrent + 10,
        keepalive_timeout=600,   # keep connections alive for the full benchmark duration
        enable_cleanup_closed=True,
    )
    completed_results: list[RequestResult] = []

    # Background task: poll the engine's server metrics periodically
    server_metrics = ServerMetrics()

    async def _poll_server_metrics():
        nonlocal server_metrics
        while True:
            try:
                server_metrics = await engine.get_server_metrics(instance)
            except Exception as e:
                logger.debug(f"Metrics poll error: {e}")
            await asyncio.sleep(metrics_poll_interval_s)

    def _attach_server_metrics_and_log(r: RequestResult, req: RequestSpec, preemptions_before: Optional[int]):
        """Attach the latest server-metrics snapshot to a completed result and
        log its status/progress. Shared by both the independent-request
        dispatch path and the multi-turn conversation dispatch path."""
        r.kv_cache_usage_pct = server_metrics.kv_cache_usage_pct
        r.num_waiting_seqs   = server_metrics.num_waiting_seqs
        r.num_running_seqs   = server_metrics.num_running_seqs
        r.num_preemptions    = server_metrics.num_preemptions_total
        r.num_preemptions_delta = (
            max(0, r.num_preemptions - preemptions_before)
            if r.num_preemptions is not None and preemptions_before is not None
            else None
        )
        r.spec_draft_acceptance_rate = server_metrics.spec_draft_acceptance_rate
        r.spec_decode_efficiency     = server_metrics.spec_decode_efficiency
        r.prefix_cache_hit_rate      = server_metrics.prefix_cache_hit_rate
        completed_results.append(r)

        n_done = len(completed_results)
        n_ok_so_far = sum(1 for x in completed_results if x.status == "ok")
        status_icon = "✓" if r.status == "ok" else "✗"
        logger.info(
            f"{status_icon} [{n_done}/{n_total}] {req.request_id} "
            f"TTFT={r.ttft_s*1000:.0f}ms E2E={r.e2e_s*1000:.0f}ms "
            f"tokens={r.output_tokens} status={r.status}"
        )
        if n_done % 10 == 0:
            n_err = n_done - n_ok_so_far
            logger.info(
                f"  Progress: {n_done}/{n_total} completed "
                f"({n_ok_so_far} ok, {n_err} errors)"
            )

    async def _dispatch(session: aiohttp.ClientSession, req: RequestSpec):
        preemptions_before = server_metrics.num_preemptions_total
        async with semaphore:
            r = await _send_request(
                session, instance, req,
                strategy_name=strategy.name,
                workload_name=workload_config.name,
                timeout_s=request_timeout_s,
                ttft_timeout_s=ttft_timeout_s,
            )
        _attach_server_metrics_and_log(r, req, preemptions_before)

    async def _dispatch_conversation(session: aiohttp.ClientSession, turns: list[RequestSpec]):
        """Replay one multi-turn conversation as sequential turns, each
        turn's prompt growing to include prior turns' text + generated
        response. The concurrency semaphore is held only during an active
        turn's request — released between turns — so waiting-for-next-turn
        time doesn't hold a concurrency slot (confirmed design choice: models
        real client behavior and avoids starving other concurrent
        conversations)."""
        prior_text = ""
        for turn_spec in turns:
            if turn_spec.turn_index == 0:
                elapsed = time.perf_counter() - trace_start
                delay = turn_spec.arrival_time - elapsed
                if delay > 0:
                    await asyncio.sleep(delay)
            else:
                await asyncio.sleep(workload_config.turn_think_time_s)
                turn_spec.prompt = f"{prior_text}\n{turn_spec.prompt}"
                turn_spec.prompt_token_len = _count_tokens(turn_spec.prompt)

            preemptions_before = server_metrics.num_preemptions_total
            async with semaphore:
                r = await _send_request(
                    session, instance, turn_spec,
                    strategy_name=strategy.name,
                    workload_name=workload_config.name,
                    timeout_s=request_timeout_s,
                    ttft_timeout_s=ttft_timeout_s,
                )
            _attach_server_metrics_and_log(r, turn_spec, preemptions_before)

            if r.status != "ok":
                break  # stop the conversation early rather than build a corrupted growing prompt
            prior_text = f"{turn_spec.prompt}\n{r.response_text}"

    trace_start = time.perf_counter()
    t_wall_start = time.time()

    async with aiohttp.ClientSession(connector=connector) as session:
        metrics_task = asyncio.create_task(_poll_server_metrics())
        tasks = []

        if is_multiturn:
            # One task per conversation — turns within a conversation are
            # dispatched sequentially inside _dispatch_conversation, not here.
            for conv in conversations:
                tasks.append(asyncio.create_task(_dispatch_conversation(session, conv)))
            logger.info(f"All {len(conversations)} conversations dispatched. Waiting for completions ...")
        else:
            for req in trace:
                # Wait until this request's scheduled arrival time
                elapsed = time.perf_counter() - trace_start
                delay = req.arrival_time - elapsed
                if delay > 0:
                    await asyncio.sleep(delay)

                tasks.append(asyncio.create_task(_dispatch(session, req)))
            logger.info(f"All {n_total} requests dispatched. Waiting for completions ...")

        results = await asyncio.gather(*tasks, return_exceptions=True)
        n_exceptions = sum(1 for r in results if isinstance(r, Exception))
        if n_exceptions:
            logger.warning(f"{n_exceptions} dispatch tasks raised exceptions")
        metrics_task.cancel()

        # Some engines (TensorRT-LLM) can't be scraped mid-run and only
        # expose a final snapshot once the server goes idle — fetch it here,
        # while the server is still up, not in engine.stop() (main.py calls
        # that after this function has already returned and saved every
        # request row, which is too late for the snapshot to reach anything).
        final_metrics = await engine.finalize_metrics(instance)
        if final_metrics is not None and completed_results:
            last = completed_results[-1]
            last.kv_cache_usage_pct = final_metrics.kv_cache_usage_pct
            last.num_waiting_seqs   = final_metrics.num_waiting_seqs
            last.num_running_seqs   = final_metrics.num_running_seqs
            last.num_preemptions    = final_metrics.num_preemptions_total
            last.spec_draft_acceptance_rate = final_metrics.spec_draft_acceptance_rate
            last.spec_decode_efficiency     = final_metrics.spec_decode_efficiency
            last.prefix_cache_hit_rate      = final_metrics.prefix_cache_hit_rate

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
