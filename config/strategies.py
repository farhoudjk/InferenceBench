"""
Execution strategy definitions for LLM inference benchmark.

Each strategy is a named configuration tuple over:
  prefill_mode       : full | chunked-N
  decode_placement   : colocated | disaggregated (disaggregated requires separate prefill instance)
  speculation        : none | ngram-K
  tensor_parallel    : 1 | 2 | 4
  quantization       : none | awq | fp8
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Strategy:
    name: str
    tensor_parallel_size: int = 1
    enable_chunked_prefill: bool = False
    chunked_prefill_size: Optional[int] = None   # kept for naming compat; unused in args
    max_num_seqs: Optional[int] = None           # max concurrent sequences in scheduler
    enable_speculation: bool = False
    speculative_model: Optional[str] = None       # path or HF id for draft model
    num_speculative_tokens: int = 5
    ngram_prompt_lookup_max: Optional[int] = None # for ngram speculation
    quantization: Optional[str] = None            # "awq" | "fp8" | None
    dtype: str = "float16"                         # "float16" | "bfloat16" | "auto"
    enforce_eager: bool = False
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192
    disaggregated_prefill: bool = False

    def to_vllm_args(self) -> list[str]:
        """Convert strategy to vLLM server CLI arguments."""
        args = [
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
            "--dtype", self.dtype,
            "--guided-decoding-backend", "lm-format-enforcer",
            "--disable-log-requests",
        ]
        if self.enforce_eager:
            args += ["--enforce-eager"]
        if self.enable_chunked_prefill:
            args += ["--enable-chunked-prefill"]
        if self.max_num_seqs is not None:
            args += ["--max-num-seqs", str(self.max_num_seqs)]
        if self.tensor_parallel_size > 1:
            # vLLM's custom all-reduce kernel runs its own independent CUDA P2P
            # probe/setup that ignores NCCL_P2P_DISABLE. On this host the two V100s
            # sit behind separate PCIe host bridges (no NVLink, no shared switch),
            # so the custom all-reduce P2P path hangs the engine step indefinitely
            # (0 tok/s, "Running" forever). Disabling it forces vLLM to fall back
            # to plain NCCL all-reduce, which works correctly over the SHM transport.
            args += ["--disable-custom-all-reduce"]

        if self.enable_speculation:
            # vLLM 0.6.0 requires the V2 block manager for any speculative decoding mode.
            args += ["--use-v2-block-manager"]
            if self.speculative_model:
                args += [
                    "--speculative-model", self.speculative_model,
                    "--num-speculative-tokens", str(self.num_speculative_tokens),
                ]
            elif self.ngram_prompt_lookup_max:
                args += [
                    "--speculative-model", "[ngram]",
                    "--ngram-prompt-lookup-max", str(self.ngram_prompt_lookup_max),
                    "--num-speculative-tokens", str(self.num_speculative_tokens),
                ]
        if self.quantization:
            args += ["--quantization", self.quantization]

        return args


# ── Strategy Space ──────────────────────────────────────────────────────────
# Fractional factorial design: covers key dimensions without full cross-product.
# Grouped by the primary dimension being varied.

STRATEGIES: dict[str, Strategy] = {

    # ── Baseline ──────────────────────────────────────────────────────────
    "baseline_full_fp16_tp1": Strategy(
        name="baseline_full_fp16_tp1",
        tensor_parallel_size=1,
        enable_chunked_prefill=False,
    ),

    # ── Scheduler Concurrency Variants (replaces chunked prefill on Volta) ────
    # --enable-chunked-prefill uses prefix_prefill.py (Triton MMA, Ampere+ only).
    # On Volta (V100, compute 7.0) it aborts at runtime. We vary max_num_seqs
    # instead: this controls how many sequences the scheduler runs in parallel,
    # which meaningfully affects TTFT, KV-cache pressure, and throughput trade-offs.
    "chunked_256_fp16_tp1": Strategy(
        name="chunked_256_fp16_tp1",
        tensor_parallel_size=1,
        chunked_prefill_size=256,   # label compat only
        max_num_seqs=16,
    ),
    "chunked_512_fp16_tp1": Strategy(
        name="chunked_512_fp16_tp1",
        tensor_parallel_size=1,
        chunked_prefill_size=512,
        max_num_seqs=32,
    ),
    "chunked_1024_fp16_tp1": Strategy(
        name="chunked_1024_fp16_tp1",
        tensor_parallel_size=1,
        chunked_prefill_size=1024,
        max_num_seqs=64,
    ),

    # ── Tensor Parallelism Variants ────────────────────────────────────────
    "baseline_full_fp16_tp2": Strategy(
        name="baseline_full_fp16_tp2",
        tensor_parallel_size=2,
        enable_chunked_prefill=False,
    ),
    "baseline_full_fp16_tp4": Strategy(
        name="baseline_full_fp16_tp4",
        tensor_parallel_size=4,
        enable_chunked_prefill=False,
    ),
    "chunked_512_fp16_tp2": Strategy(
        name="chunked_512_fp16_tp2",
        tensor_parallel_size=2,
        chunked_prefill_size=512,
        max_num_seqs=32,
    ),

    # ── Speculative Decoding: ngram ────────────────────────────────────────
    "spec_ngram3_fp16_tp1": Strategy(
        name="spec_ngram3_fp16_tp1",
        tensor_parallel_size=1,
        enable_speculation=True,
        ngram_prompt_lookup_max=3,
        num_speculative_tokens=5,
    ),
    "spec_ngram5_fp16_tp1": Strategy(
        name="spec_ngram5_fp16_tp1",
        tensor_parallel_size=1,
        enable_speculation=True,
        ngram_prompt_lookup_max=5,
        num_speculative_tokens=5,
    ),

    # ── Quantization Variants ──────────────────────────────────────────────
    "baseline_full_fp8_tp1": Strategy(
        name="baseline_full_fp8_tp1",
        tensor_parallel_size=1,
        enable_chunked_prefill=False,
        quantization="fp8",
    ),
    "chunked_512_fp8_tp1": Strategy(
        name="chunked_512_fp8_tp1",
        tensor_parallel_size=1,
        chunked_prefill_size=512,
        max_num_seqs=32,
        quantization="fp8",
    ),

    # ── Interaction: Chunked + Speculation ────────────────────────────────
    "chunked_512_spec_ngram3_fp16_tp1": Strategy(
        name="chunked_512_spec_ngram3_fp16_tp1",
        tensor_parallel_size=1,
        chunked_prefill_size=512,
        max_num_seqs=32,
        enable_speculation=True,
        ngram_prompt_lookup_max=3,
        num_speculative_tokens=5,
    ),
}


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]


def get_strategies_for_dimension(dimension: str) -> dict[str, Strategy]:
    """Return strategy subset for ablation over a single dimension."""
    groups = {
        "chunked_prefill": [
            "baseline_full_fp16_tp1",
            "chunked_256_fp16_tp1",
            "chunked_512_fp16_tp1",
            "chunked_1024_fp16_tp1",
        ],
        "tensor_parallel": [
            "baseline_full_fp16_tp1",
            "baseline_full_fp16_tp2",
            "baseline_full_fp16_tp4",
        ],
        "speculation": [
            "baseline_full_fp16_tp1",
            "spec_ngram3_fp16_tp1",
            "spec_ngram5_fp16_tp1",
        ],
        "quantization": [
            "baseline_full_fp16_tp1",
            "baseline_full_fp8_tp1",
        ],
        "interaction": [
            "baseline_full_fp16_tp1",
            "chunked_512_fp16_tp1",
            "spec_ngram3_fp16_tp1",
            "chunked_512_spec_ngram3_fp16_tp1",
        ],
    }
    if dimension not in groups:
        raise ValueError(f"Unknown dimension '{dimension}'. Choose from: {list(groups.keys())}")
    return {k: STRATEGIES[k] for k in groups[dimension]}
