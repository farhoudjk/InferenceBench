"""
SGLang engine adapter.

Flag names and default behavior verified against the installed SGLang
version's `python -m sglang.launch_server --help` (SGLang's CLI is close
enough to vLLM's that most Strategy fields map directly, but its chunked-
prefill and speculative-decoding knobs differ semantically — see the
best-effort mapping notes below). Metrics verified against a live instance's
/metrics output with --enable-metrics.
"""

import logging
from typing import Optional

import aiohttp

from benchmark.engines.base import Engine, EngineInstance, ServerMetrics
from config.strategies import Strategy

logger = logging.getLogger(__name__)


class SGLangEngine(Engine):
    name = "sglang"

    def module_name(self) -> str:
        return "sglang.launch_server"

    def build_args(self, strategy: Strategy, port: int) -> list[str]:
        args = [
            "--model-path", self.model,
            "--port", str(port),
            "--host", "127.0.0.1",
            "--tensor-parallel-size", str(strategy.tensor_parallel_size),
            "--mem-fraction-static", str(strategy.gpu_memory_utilization),
            "--context-length", str(strategy.max_model_len),
            "--dtype", strategy.dtype,
            "--enable-metrics",
        ]

        # SGLang's chunked-prefill token budget is the closest equivalent to
        # vLLM's --max-num-batched-tokens, but the two engines may not chunk
        # identically internally — cross-engine chunked-prefill comparisons
        # should be read as "each engine's own interpretation of chunk size,"
        # not a verified identical mechanism.
        if strategy.max_num_batched_tokens is not None:
            args += ["--chunked-prefill-size", str(strategy.max_num_batched_tokens)]
        if strategy.max_num_seqs is not None:
            args += ["--max-running-requests", str(strategy.max_num_seqs)]
        if strategy.quantization:
            args += ["--quantization", strategy.quantization]
        if not strategy.enable_prefix_caching:
            # SGLang's RadixAttention prefix caching is on by default with no
            # positive "--enable-radix-cache" flag to pass — verified via
            # --help, only --disable-radix-cache exists. So the on case is a
            # no-op here, matching SGLang's own default (same asymmetry as
            # not needing to pass anything for vLLM's other always-on features).
            args += ["--disable-radix-cache"]

        if strategy.enable_speculation:
            if strategy.ngram_prompt_lookup_max:
                # SGLang's ngram speculative decoding has no direct
                # "prompt_lookup_max" knob — it's configured via match-window/
                # BFS-breadth/capacity parameters with different semantics
                # than vLLM's prompt_lookup_max. Map only num_speculative_tokens
                # (the one field with a clear equivalent) and warn that the
                # rest of the ngram config is left at SGLang's defaults.
                args += [
                    "--speculative-algorithm", "NGRAM",
                    "--speculative-num-draft-tokens", str(strategy.num_speculative_tokens),
                ]
                logger.warning(
                    f"[{strategy.name}] SGLang's ngram speculative decoding has no "
                    f"equivalent to ngram_prompt_lookup_max={strategy.ngram_prompt_lookup_max}; "
                    "using SGLang's default ngram match-window/capacity settings. "
                    "Cross-engine ngram comparisons reflect each engine's own "
                    "ngram configuration, not an identical mechanism."
                )
            elif strategy.speculative_model:
                args += [
                    "--speculative-algorithm", "EAGLE",
                    "--speculative-draft-model-path", strategy.speculative_model,
                    "--speculative-num-draft-tokens", str(strategy.num_speculative_tokens),
                ]

        return args

    @classmethod
    def process_match_patterns(cls) -> list[str]:
        return ["sglang.launch_server"]

    async def get_server_metrics(self, instance: EngineInstance) -> ServerMetrics:
        """Fetch Prometheus metrics from SGLang's /metrics endpoint (requires
        the server to have been started with --enable-metrics, which
        build_args() always adds).

        SGLang's metric lines carry Prometheus labels (e.g.
        `sglang:num_running_reqs{model_name="...",tp_rank="0"} 0.0`), unlike
        vLLM's unlabeled gauge lines — match on `name{` / `name ` rather than
        a plain prefix, and only take the first matching series (TP rank 0)
        rather than summing across ranks.

        sglang:cache_hit_rate is SGLang's own prefix-cache reuse gauge
        (0.0-1.0, direct equivalent of vLLM's derived prefix_cache_hit_rate) —
        confirmed present in this installed version via a live /metrics
        scrape after sending two overlapping-prefix requests.
        """
        target_metrics = {
            "sglang:token_usage",
            "sglang:num_running_reqs",
            "sglang:num_queue_reqs",
            "sglang:cache_hit_rate",
        }

        raw: dict = {}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    instance.metrics_url, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status != 200:
                        return ServerMetrics()
                    text = await resp.text()

            for line in text.splitlines():
                if line.startswith("#"):
                    continue
                for key in target_metrics:
                    if key in raw:
                        continue  # keep only the first (TP rank 0) series
                    if line.startswith(key + "{") or line.startswith(key + " "):
                        parts = line.rsplit(" ", 1)
                        if len(parts) == 2:
                            try:
                                raw[key] = float(parts[1])
                            except ValueError:
                                pass
        except Exception:
            return ServerMetrics()

        return ServerMetrics(
            kv_cache_usage_pct=raw.get("sglang:token_usage"),
            num_waiting_seqs=_as_int(raw.get("sglang:num_queue_reqs")),
            num_running_seqs=_as_int(raw.get("sglang:num_running_reqs")),
            num_preemptions_total=None,   # no equivalent counter exposed by SGLang
            spec_draft_acceptance_rate=None,  # no direct gauge; not verified against a spec-decode run
            spec_decode_efficiency=None,
            # sglang:cache_hit_rate is already a ratio (gauge), unlike vLLM's
            # counter-pair-derived rate — verified live (98.5% after two
            # overlapping-prefix requests) via `sglang.launch_server
            # --enable-metrics` + a real /metrics scrape.
            prefix_cache_hit_rate=raw.get("sglang:cache_hit_rate"),
            raw=raw,
        )


def _as_int(value: Optional[float]) -> Optional[int]:
    return int(value) if value is not None else None
