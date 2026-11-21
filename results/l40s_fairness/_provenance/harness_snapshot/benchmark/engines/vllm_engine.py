"""
vLLM engine adapter.

Owns everything specific to vLLM: CLI flag translation from Strategy, the
hardware workarounds this host needed (V100 enforce-eager, AWQ compute-
capability gate, flashinfer sampler JIT disable, NCCL/Gloo topology pinning),
and Prometheus /metrics scraping into the normalized ServerMetrics schema.
"""

import json
import logging
from typing import Optional

import aiohttp

from benchmark.engines.base import Engine, EngineInstance, ServerMetrics
from config.strategies import Strategy

logger = logging.getLogger(__name__)


class VLLMEngine(Engine):
    name = "vllm"

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

    # AWQ kernels require Turing or newer (compute capability >= 7.5).
    # V100 (Volta) is 7.0 — vLLM builds and starts loading the model before
    # discovering this, wasting ~15-25s per attempt with a deep traceback.
    _AWQ_MIN_COMPUTE_CAPABILITY = (7, 5)

    def _needs_enforce_eager(self) -> bool:
        model_lower = self.model.lower()
        return any(k in model_lower for k in self._ENFORCE_EAGER_MODELS)

    def _native_max_model_len(self) -> Optional[int]:
        model_lower = self.model.lower()
        for key, limit in self._MAX_MODEL_LEN_OVERRIDES.items():
            if key in model_lower:
                return limit
        return None

    def preflight_check(self, strategy: Strategy) -> None:
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

    def module_name(self) -> str:
        return "vllm.entrypoints.openai.api_server"

    def build_args(self, strategy: Strategy, port: int) -> list[str]:
        native_limit = self._native_max_model_len()
        if native_limit is not None and strategy.max_model_len > native_limit:
            logger.info(
                f"Capping max_model_len for model {self.model}: "
                f"{strategy.max_model_len} → {native_limit} (model's native context limit)"
            )
            strategy.max_model_len = native_limit

        args = [
            "--model", self.model,
            "--port", str(port),
            "--host", "127.0.0.1",
            "--tensor-parallel-size", str(strategy.tensor_parallel_size),
            "--gpu-memory-utilization", str(strategy.gpu_memory_utilization),
            "--max-model-len", str(strategy.max_model_len),
            "--dtype", strategy.dtype,
            "--guided-decoding-backend", "lm-format-enforcer",
            "--disable-log-requests",
        ]
        if strategy.enforce_eager:
            args += ["--enforce-eager"]
        if strategy.enable_chunked_prefill:
            args += ["--enable-chunked-prefill"]
        # This vLLM build defaults automatic prefix caching to on with no flag
        # passed at all — pass it explicitly either way so every run's logged
        # command line states the condition instead of relying on an unstated
        # version-specific default (verified via `--help`: both
        # --enable-prefix-caching and --no-enable-prefix-caching exist here).
        args += ["--enable-prefix-caching" if strategy.enable_prefix_caching else "--no-enable-prefix-caching"]
        if strategy.max_num_batched_tokens is not None:
            args += ["--max-num-batched-tokens", str(strategy.max_num_batched_tokens)]
        if strategy.max_num_seqs is not None:
            args += ["--max-num-seqs", str(strategy.max_num_seqs)]
        if strategy.tensor_parallel_size > 1:
            # vLLM's custom all-reduce kernel runs its own independent CUDA P2P
            # probe/setup that ignores NCCL_P2P_DISABLE. On this host the two V100s
            # sit behind separate PCIe host bridges (no NVLink, no shared switch),
            # so the custom all-reduce P2P path hangs the engine step indefinitely
            # (0 tok/s, "Running" forever). Disabling it forces vLLM to fall back
            # to plain NCCL all-reduce, which works correctly over the SHM transport.
            args += ["--disable-custom-all-reduce"]

        if strategy.enable_speculation:
            # vLLM >=0.9 collapsed --speculative-model/--use-v2-block-manager/
            # --ngram-prompt-lookup-max into one --speculative-config JSON dict;
            # the V1 engine has no block-manager selector at all.
            if strategy.speculative_model:
                spec_config = {
                    "model": strategy.speculative_model,
                    "num_speculative_tokens": strategy.num_speculative_tokens,
                }
            elif strategy.ngram_prompt_lookup_max:
                spec_config = {
                    "method": "ngram",
                    "num_speculative_tokens": strategy.num_speculative_tokens,
                    "prompt_lookup_max": strategy.ngram_prompt_lookup_max,
                }
            else:
                spec_config = None
            if spec_config is not None:
                args += ["--speculative-config", json.dumps(spec_config)]
        if strategy.quantization:
            args += ["--quantization", strategy.quantization]

        # Some models' CUDA graph warmup hangs on V100; fall back to eager execution.
        if self._needs_enforce_eager() and "--enforce-eager" not in args:
            args += ["--enforce-eager"]
            logger.info(f"Auto-adding --enforce-eager for model {self.model} (V100 CUDA graph workaround)")

        return args

    def env_overrides(self, strategy: Strategy) -> dict[str, str]:
        env: dict[str, str] = {}
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

        return env

    @classmethod
    def process_match_patterns(cls) -> list[str]:
        return ["vllm.entrypoints.openai.api_server"]

    async def get_server_metrics(self, instance: EngineInstance) -> ServerMetrics:
        """Fetch Prometheus metrics from vLLM's /metrics endpoint.

        Metric names verified against vLLM 0.11.0's actual V1 engine output
        (vllm/v1/metrics/loggers.py, vllm/v1/spec_decode/metrics.py) — several
        differ from older vLLM versions' names (e.g. num_waiting_seqs ->
        num_requests_waiting; num_preemptions_total -> num_preemptions, which
        the Prometheus client auto-suffixes to _total for Counter types in the
        exposed text). vLLM has no direct spec-decode acceptance-rate gauge;
        it's derived here from the cumulative accepted/draft token counters.

        vllm:prefix_cache_queries/vllm:prefix_cache_hits are the current,
        non-deprecated counter names (verified in source) — a same-named
        vllm:gpu_prefix_cache_queries/hits pair also exists but is deprecated
        and hidden behind --show-hidden-metrics-for-version, not emitted by
        default, so no risk of the startswith() match below picking up the
        wrong one under default config.
        """
        target_metrics = {
            "vllm:kv_cache_usage_perc",
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
            "vllm:num_preemptions_total",
            "vllm:prompt_tokens_total",
            "vllm:generation_tokens_total",
            "vllm:spec_decode_num_accepted_tokens_total",
            "vllm:spec_decode_num_draft_tokens_total",
            "vllm:prefix_cache_queries_total",
            "vllm:prefix_cache_hits_total",
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
                    if line.startswith(key):
                        parts = line.rsplit(" ", 1)
                        if len(parts) == 2:
                            try:
                                raw[key] = float(parts[1])
                            except ValueError:
                                pass
        except Exception:
            return ServerMetrics()

        accepted = raw.get("vllm:spec_decode_num_accepted_tokens_total")
        drafted = raw.get("vllm:spec_decode_num_draft_tokens_total")
        acceptance_rate = accepted / drafted if accepted is not None and drafted else None

        cache_hits = raw.get("vllm:prefix_cache_hits_total")
        cache_queries = raw.get("vllm:prefix_cache_queries_total")
        prefix_cache_hit_rate = (
            cache_hits / cache_queries if cache_hits is not None and cache_queries else None
        )

        return ServerMetrics(
            kv_cache_usage_pct=raw.get("vllm:kv_cache_usage_perc"),
            num_waiting_seqs=_as_int(raw.get("vllm:num_requests_waiting")),
            num_running_seqs=_as_int(raw.get("vllm:num_requests_running")),
            num_preemptions_total=_as_int(raw.get("vllm:num_preemptions_total")),
            spec_draft_acceptance_rate=acceptance_rate,
            spec_decode_efficiency=None,  # no direct vLLM 0.11.0 gauge for this
            prefix_cache_hit_rate=prefix_cache_hit_rate,
            raw=raw,
        )


def _as_int(value: Optional[float]) -> Optional[int]:
    return int(value) if value is not None else None
