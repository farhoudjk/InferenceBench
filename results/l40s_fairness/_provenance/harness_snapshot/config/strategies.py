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
    """Engine-agnostic scheduling-config: what to test, not how to spell it
    for a given engine. CLI arg translation lives in each Engine subclass's
    build_args() (benchmark/engines/{vllm,sglang}_engine.py) — the same
    Strategy object drives either engine."""
    name: str
    tensor_parallel_size: int = 1
    enable_chunked_prefill: bool = False
    chunked_prefill_size: Optional[int] = None   # kept for naming compat; unused in args
    max_num_batched_tokens: Optional[int] = None # per-step token budget = chunk size
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
    enable_prefix_caching: bool = True  # vLLM/SGLang default this on; RQ2 needs an explicit off toggle


# ── Strategy Space ──────────────────────────────────────────────────────────
# Fractional factorial design: covers key dimensions without full cross-product.
# Grouped by the primary dimension being varied.

STRATEGIES: dict[str, Strategy] = {

    # ── Baseline ──────────────────────────────────────────────────────────
    # vLLM 0.11.0's V1 scheduler always chunks prefill (no "off" switch) — the
    # --enable-chunked-prefill flag is a no-op there. The unchunked-prefill
    # equivalent is a token budget >= max_model_len, so any prompt fits in one
    # scheduler step. max_num_seqs is pinned to 256 here and matched across
    # every chunked_* variant below, so those sweeps vary only chunk size.
    "baseline_full_fp16_tp1": Strategy(
        name="baseline_full_fp16_tp1",
        tensor_parallel_size=1,
        enable_chunked_prefill=False,
        max_num_batched_tokens=8192,
        max_num_seqs=256,
    ),

    # ── Chunked Prefill Variants (max_num_batched_tokens = chunk size) ────────
    # max_num_seqs is held constant at 256 (matching baseline) so these sweeps
    # vary only the per-step token budget, not admission concurrency.
    # enable_chunked_prefill=True is a no-op for vLLM (V1 scheduler always
    # chunks) and for SGLang (--chunked-prefill-size applies regardless of any
    # separate enable flag), but it is load-bearing for TensorRT-LLM: without
    # it, max_num_batched_tokens becomes a hard per-request prompt-length
    # ceiling (HTTP 400 above that length) rather than a chunking budget —
    # confirmed live via RQ2's chat_real_long sweep, where omitting this flag
    # caused a 99.3% failure rate once prompts exceeded the "chunk" size.
    "chunked_256_fp16_tp1": Strategy(
        name="chunked_256_fp16_tp1",
        tensor_parallel_size=1,
        enable_chunked_prefill=True,
        chunked_prefill_size=256,   # label compat only
        max_num_batched_tokens=256,
        max_num_seqs=256,
    ),
    "chunked_512_fp16_tp1": Strategy(
        name="chunked_512_fp16_tp1",
        tensor_parallel_size=1,
        enable_chunked_prefill=True,
        chunked_prefill_size=512,
        max_num_batched_tokens=512,
        max_num_seqs=256,
    ),
    "chunked_1024_fp16_tp1": Strategy(
        name="chunked_1024_fp16_tp1",
        tensor_parallel_size=1,
        enable_chunked_prefill=True,
        chunked_prefill_size=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=256,
    ),

    # ── Tensor Parallelism Variants ────────────────────────────────────────
    "baseline_full_fp16_tp2": Strategy(
        name="baseline_full_fp16_tp2",
        tensor_parallel_size=2,
        enable_chunked_prefill=False,
        max_num_batched_tokens=8192,
        max_num_seqs=256,
    ),
    "baseline_full_fp16_tp4": Strategy(
        name="baseline_full_fp16_tp4",
        tensor_parallel_size=4,
        enable_chunked_prefill=False,
        max_num_batched_tokens=8192,
        max_num_seqs=256,
    ),
    "chunked_512_fp16_tp2": Strategy(
        name="chunked_512_fp16_tp2",
        tensor_parallel_size=2,
        enable_chunked_prefill=True,
        chunked_prefill_size=512,
        max_num_batched_tokens=512,
        max_num_seqs=256,
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
        max_num_batched_tokens=8192,
        max_num_seqs=256,
        quantization="fp8",
    ),
    "chunked_512_fp8_tp1": Strategy(
        name="chunked_512_fp8_tp1",
        tensor_parallel_size=1,
        enable_chunked_prefill=True,
        chunked_prefill_size=512,
        max_num_batched_tokens=512,
        max_num_seqs=256,
        quantization="fp8",
    ),

    # ── Interaction: Chunked + Speculation ────────────────────────────────
    "chunked_512_spec_ngram3_fp16_tp1": Strategy(
        name="chunked_512_spec_ngram3_fp16_tp1",
        tensor_parallel_size=1,
        enable_chunked_prefill=True,
        chunked_prefill_size=512,
        max_num_batched_tokens=512,
        max_num_seqs=256,
        enable_speculation=True,
        ngram_prompt_lookup_max=3,
        num_speculative_tokens=5,
    ),

    # ── RQ2: Prefill-handling strategies ───────────────────────────────────
    # Chunk size x prefix-caching, crossed. The chunked_*_fp16_tp1 strategies
    # above already cover prefix-caching=True (the dataclass default) at each
    # chunk size, so only the prefix-caching=False leg needs new entries here
    # — 4 new strategies cover the missing half of a 4 (chunk sizes: none/
    # 256/512/1024) x 2 (prefix caching on/off) factorial design.
    "baseline_full_fp16_tp1_nopfx": Strategy(
        name="baseline_full_fp16_tp1_nopfx",
        tensor_parallel_size=1,
        enable_chunked_prefill=False,
        max_num_batched_tokens=8192,
        max_num_seqs=256,
        enable_prefix_caching=False,
    ),
    "chunked_256_fp16_tp1_nopfx": Strategy(
        name="chunked_256_fp16_tp1_nopfx",
        tensor_parallel_size=1,
        enable_chunked_prefill=True,
        chunked_prefill_size=256,
        max_num_batched_tokens=256,
        max_num_seqs=256,
        enable_prefix_caching=False,
    ),
    "chunked_512_fp16_tp1_nopfx": Strategy(
        name="chunked_512_fp16_tp1_nopfx",
        tensor_parallel_size=1,
        enable_chunked_prefill=True,
        chunked_prefill_size=512,
        max_num_batched_tokens=512,
        max_num_seqs=256,
        enable_prefix_caching=False,
    ),
    "chunked_1024_fp16_tp1_nopfx": Strategy(
        name="chunked_1024_fp16_tp1_nopfx",
        tensor_parallel_size=1,
        enable_chunked_prefill=True,
        chunked_prefill_size=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=256,
        enable_prefix_caching=False,
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
        "prefill_handling": [
            "baseline_full_fp16_tp1",
            "baseline_full_fp16_tp1_nopfx",
            "chunked_256_fp16_tp1",
            "chunked_256_fp16_tp1_nopfx",
            "chunked_512_fp16_tp1",
            "chunked_512_fp16_tp1_nopfx",
            "chunked_1024_fp16_tp1",
            "chunked_1024_fp16_tp1_nopfx",
        ],
    }
    if dimension not in groups:
        raise ValueError(f"Unknown dimension '{dimension}'. Choose from: {list(groups.keys())}")
    return {k: STRATEGIES[k] for k in groups[dimension]}
