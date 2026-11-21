"""
TensorRT-LLM engine adapter.

Runs in its own venv (.venv-trtllm), not the shared one vLLM/SGLang live in:
tensorrt-llm's pinned deps (torch, transformers, flashinfer, xgrammar,
nvidia-cutlass-dsl) conflict with both vLLM 0.11.0's and SGLang 0.5.5's pins
closely enough that installing it into the shared venv breaks vLLM's compiled
CUDA extension (vllm._C fails to load — verified on this host). Point
TRTLLM_VENV at that separate venv's root; python_executable() resolves its
interpreter.

Also: tensorrt-llm's PyPI releases >=1.1.0 are built against CUDA 13
(nvidia-nccl-cu13, cuda-python>=13) and fail with "CUDA driver version is
insufficient" on drivers whose max supported CUDA version is 12.8 (check via
`nvidia-smi`). Pin tensorrt-llm==1.0.0 (last CUDA-12 release) in that venv on
such hosts.

CLI verified against a live `trtllm-serve serve --help` (v1.0.0) and a live
/metrics scrape from an actual served request (Llama-3.1-8B-Instruct,
--backend pytorch) on this host. trtllm-serve's CLI is much thinner than
vLLM/SGLang's: most Strategy knobs (dtype, quantization, chunked-prefill,
speculative decoding) have no direct flag in v1.0.0 and must go through
--extra_llm_api_options, a YAML file of LlmArgs fields — build_args() writes
one per launch into results_dir.
"""

import logging
import os
from typing import Optional

import requests
import yaml

from benchmark.engines.base import Engine, EngineInstance, ServerMetrics
from config.strategies import Strategy

logger = logging.getLogger(__name__)

# Root of tensorrt-llm's dedicated venv. Override via env var if it lives
# elsewhere; see this module's docstring for why it's not the shared venv.
TRTLLM_VENV = os.environ.get(
    "TRTLLM_VENV",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".venv-trtllm"),
)


class TensorRTLLMEngine(Engine):
    name = "tensorrtllm"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_metrics: Optional[ServerMetrics] = None

    def python_executable(self) -> str:
        python_path = os.path.join(TRTLLM_VENV, "bin", "python3")
        if not os.path.exists(python_path):
            raise RuntimeError(
                f"TensorRT-LLM venv not found at {TRTLLM_VENV!r}. Create it separately "
                f"from the main venv (its pinned deps conflict with vLLM/SGLang's) via:\n"
                f"  python3 -m venv {TRTLLM_VENV} && {TRTLLM_VENV}/bin/pip install tensorrt-llm==1.0.0\n"
                f"(use tensorrt-llm==1.0.0 specifically if your driver's max CUDA version is 12.8 — "
                f"see this module's docstring). Override the path via the TRTLLM_VENV env var."
            )
        return python_path

    def preflight_check(self, strategy: Strategy) -> None:
        self._last_metrics = None  # don't leak a previous instance's metrics into this one
        self.python_executable()  # raises with a clear message if the venv is missing
        if strategy.disaggregated_prefill:
            raise RuntimeError(
                "disaggregated_prefill=True is not implemented for TensorRTLLMEngine "
                "(trtllm-serve's disaggregated mode needs a separate context/generation "
                "server pair, not a single-process flag — out of scope for this adapter)."
            )

    def env_overrides(self, strategy: Strategy) -> dict[str, str]:
        # A venv's own bin/ is only on PATH when the venv is *activated*
        # (`source .venv-trtllm/bin/activate`) — launching its interpreter by
        # absolute path via subprocess.Popen (as python_executable() does)
        # does not add it. flashinfer JIT-compiles some kernels (e.g. RMSNorm)
        # on first use via `ninja`, which then isn't found on PATH and crashes
        # the executor worker with "Ninja is required to load C++ extensions"
        # (verified live on this host). Prepend the venv's bin/ so its ninja
        # resolves; nvcc is assumed to be on system PATH already (this host
        # has nvidia-cuda-toolkit installed via apt for SGLang's equivalent
        # flashinfer JIT-compile needs — see sglang_engine.py's history).
        venv_bin = os.path.join(TRTLLM_VENV, "bin")
        return {"PATH": f"{venv_bin}:{os.environ.get('PATH', '')}"}

    def _launch_cmd(self, strategy: Strategy, port: int) -> list[str]:
        trtllm_serve = os.path.join(TRTLLM_VENV, "bin", "trtllm-serve")
        return [trtllm_serve, "serve", self.model] + self.build_args(strategy, port)

    def build_args(self, strategy: Strategy, port: int) -> list[str]:
        llm_api_options = self._build_llm_api_options(strategy)
        config_path = os.path.join(
            self.results_dir, f"trtllm_config_{strategy.name}_{port}.yaml"
        )
        os.makedirs(self.results_dir, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.safe_dump(llm_api_options, f)

        args = [
            "--host", "127.0.0.1",
            "--port", str(port),
            "--backend", "pytorch",   # cpp path (the CLI default) needs a pre-built
                                      # TensorRT engine dir, not a raw HF checkpoint
            "--tp_size", str(strategy.tensor_parallel_size),
            "--kv_cache_free_gpu_memory_fraction", str(strategy.gpu_memory_utilization),
            "--extra_llm_api_options", config_path,
        ]
        if strategy.max_num_seqs is not None:
            args += ["--max_batch_size", str(strategy.max_num_seqs)]
        if strategy.max_num_batched_tokens is not None:
            args += ["--max_num_tokens", str(strategy.max_num_batched_tokens)]
        args += ["--max_seq_len", str(strategy.max_model_len)]

        return args

    def _build_llm_api_options(self, strategy: Strategy) -> dict:
        """Fields with no direct trtllm-serve CLI flag (v1.0.0) — passed via
        --extra_llm_api_options YAML instead (tensorrt_llm.llmapi.llm_args.TorchLlmArgs)."""
        options: dict = {
            "enable_iter_perf_stats": True,   # required for /metrics to return non-empty stats
        }

        if strategy.dtype and strategy.dtype != "auto":
            options["dtype"] = strategy.dtype

        if strategy.enable_chunked_prefill:
            options["enable_chunked_prefill"] = True

        # KvCacheConfig.enable_block_reuse (verified via TorchLlmArgs.model_fields
        # on this installed TensorRT-LLM 1.0.0: default True) is this engine's
        # real prefix-cache/KV-block-reuse toggle — the same mechanism whose
        # cacheHitRate this adapter already reads in _store_metrics(). Passed
        # explicitly both ways so every run's YAML states the condition.
        options["kv_cache_config"] = {"enable_block_reuse": strategy.enable_prefix_caching}

        if strategy.quantization:
            quant_algo = {"awq": "W4A16_AWQ", "fp8": "FP8"}.get(strategy.quantization)
            if quant_algo is None:
                logger.warning(
                    f"[{strategy.name}] No known TensorRT-LLM QuantAlgo mapping for "
                    f"quantization={strategy.quantization!r}; leaving quant_config unset."
                )
            else:
                options["quant_config"] = {"quant_algo": quant_algo}

        if strategy.enable_speculation:
            if strategy.speculative_model:
                options["speculative_config"] = {
                    "decoding_type": "Eagle",
                    "speculative_model_dir": strategy.speculative_model,
                    "max_draft_len": strategy.num_speculative_tokens,
                }
            elif strategy.ngram_prompt_lookup_max:
                # NGramDecodingConfig has no field equivalent to vLLM's
                # prompt_lookup_max beyond max_matching_ngram_size — mapped
                # directly, but the rest of NGram's pool/matching behavior
                # (is_keep_all/is_use_oldest/is_public_pool) is left at
                # TensorRT-LLM's defaults, same caveat SGLang's adapter notes
                # for its own ngram mapping.
                options["speculative_config"] = {
                    "decoding_type": "NGram",
                    "max_draft_len": strategy.num_speculative_tokens,
                    "max_matching_ngram_size": strategy.ngram_prompt_lookup_max,
                }
                # NGram decoding asserts against TorchLlmArgs' overlap scheduler
                # (default on) at engine build time — confirmed live:
                # "AssertionError: NGram does not support overlap scheduler",
                # crashing the executor worker before it ever serves a request.
                # SGLang's server auto-disables its own overlap scheduler for
                # ngram (logged as a WARNING at startup); TensorRT-LLM has no
                # such auto-fallback, so this adapter must set it explicitly.
                options["disable_overlap_scheduler"] = True
                logger.warning(
                    f"[{strategy.name}] TensorRT-LLM's ngram speculative decoding config "
                    "differs in shape from vLLM/SGLang's; only max_draft_len and "
                    "max_matching_ngram_size are mapped, rest left at TensorRT-LLM's "
                    "defaults. Cross-engine ngram comparisons reflect each engine's own "
                    "ngram configuration, not an identical mechanism."
                )

        # Opt-in extra LlmArgs (fairness experiments): JSON in env var
        # TRTLLM_EXTRA_LLM_API_OPTIONS, deep-merged over the options above.
        # Unset (the default) -> no change to any run's YAML.
        extra = os.environ.get("TRTLLM_EXTRA_LLM_API_OPTIONS")
        if extra:
            import json
            def _merge(dst, src):
                for k, v in src.items():
                    if isinstance(v, dict) and isinstance(dst.get(k), dict):
                        _merge(dst[k], v)
                    else:
                        dst[k] = v
            _merge(options, json.loads(extra))
            logger.info(f"[{strategy.name}] TRTLLM_EXTRA_LLM_API_OPTIONS merged: {extra}")

        return options

    @classmethod
    def process_match_patterns(cls) -> list[str]:
        return ["trtllm-serve"]

    async def finalize_metrics(self, instance: EngineInstance) -> Optional[ServerMetrics]:
        """One-shot /metrics fetch. Called by run_benchmark() right after its
        request loop finishes but before engine.stop() tears the server down
        — the server is idle at that point but still reachable, which this
        fetch depends on (see docstring below). Must NOT be deferred to
        stop(): main.py calls run_benchmark() (which packages/saves every
        request row) before engine.stop(), so a fetch made only in stop()
        arrives too late to ever be attached to a request or the summary —
        confirmed live (total_preemptions/kv_cache_usage_pct came back
        null/NaN on every row of a 21-request smoke test before this was
        fixed to call finalize_metrics() instead).

        TensorRT-LLM's /metrics is not a point-in-time scrape like vLLM/
        SGLang's — it's a long-poll (`async for stat in
        self.llm.get_stats_async(2)` server-side) that only returns once the
        server has gone 2 full seconds without a new iteration. Verified live:
        during an active benchmark the server never goes idle for 2s, so any
        call made mid-run blocks for the ~entire remaining run duration (seen:
        a call issued 2s into a 17s benchmark didn't return until ~19s had
        elapsed — i.e., effectively until the run ended anyway). That makes
        per-request mid-run metrics attribution structurally unavailable for
        this engine, not just hard to get right. This single post-run fetch
        gets a real final snapshot instead (the server is idle by the time
        the request loop ends, so it returns immediately) — an aggregate/
        last-state value, not a per-request time series like vLLM/SGLang
        provide.
        """
        try:
            resp = requests.get(instance.metrics_url, timeout=10)
            if resp.status_code == 200:
                stats = resp.json()
                if stats:
                    self._store_metrics(stats[-1])
                    return self._last_metrics
        except Exception as e:
            logger.debug(f"TensorRT-LLM final metrics fetch failed: {type(e).__name__}: {e}")
        return None

    def _store_metrics(self, latest: dict) -> None:
        kv = latest.get("kvCacheStats") or {}
        max_blocks = kv.get("maxNumBlocks")
        used_blocks = kv.get("usedNumBlocks")
        kv_usage_pct = 100.0 * used_blocks / max_blocks if max_blocks else None

        self._last_metrics = ServerMetrics(
            kv_cache_usage_pct=kv_usage_pct,
            num_waiting_seqs=_as_int(latest.get("numQueuedRequests")),
            num_running_seqs=_as_int(latest.get("numActiveRequests")),
            # inflightBatchingStats.numPausedRequests is TensorRT-LLM's
            # equivalent of vLLM's vllm:num_preemptions_total — verified live
            # via a probe server: present as an int (0 in a light 15-request
            # smoke test that never had to preempt) in every /metrics sample.
            # This is a final snapshot value like the rest of this engine's
            # metrics (see finalize_metrics()'s docstring), not a per-request
            # delta — comparable to vLLM's own totals only at the run-summary
            # level, not per-request.
            num_preemptions_total=_as_int((latest.get("inflightBatchingStats") or {}).get("numPausedRequests")),
            spec_draft_acceptance_rate=None,  # specDecodingStats present but not verified
                                               # against a live spec-decode run
            spec_decode_efficiency=None,
            # kvCacheStats.cacheHitRate is a real, direct equivalent of vLLM's
            # derived prefix_cache_hit_rate (like SGLang's cache_hit_rate, an
            # already-a-ratio gauge, not a counter pair) — verified live via a
            # probe server: reusedBlocks/cacheHitRate both moved (0 -> 1 block,
            # 0.0 -> 0.00032) after a second request shared a prompt prefix
            # with the first. Missed when this adapter was first written even
            # though maxNumBlocks/usedNumBlocks from the same kvCacheStats
            # block were already being read.
            prefix_cache_hit_rate=kv.get("cacheHitRate"),
            raw=latest,
        )

    async def get_server_metrics(self, instance: EngineInstance) -> ServerMetrics:
        """Returns the last snapshot fetched by finalize_metrics() (called by
        run_benchmark() right after its request loop finishes, before
        engine.stop() tears the server down) — see finalize_metrics()'s
        docstring for why this engine can't support mid-run polling the way
        vLLM/SGLang do. Blank (ServerMetrics()) for the whole run until
        finalize_metrics() runs."""
        return self._last_metrics or ServerMetrics()


def _as_int(value: Optional[float]) -> Optional[int]:
    return int(value) if value is not None else None
