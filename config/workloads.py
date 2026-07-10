"""
Workload configuration and request trace generation.

Covers four task types x five prompt lengths x four output lengths, plus five
real-data workloads (chat_real from ShareGPT, summarization_real from
CNN/DailyMail, reasoning_real from OpenMathReasoning, code_real from MBPP,
and mixed_real which interleaves all four with each request tagged by its
real task type) for workloads that need non-repetitive, natural-language
prompts — e.g. to get realistic speculative-decoding draft acceptance rates
instead of the ~100% rate synthetic filler-repeated prompts produce.

Arrival processes: Poisson (low/med/high rps), bursty.

Rate is decoupled from workload shape: pick a WorkloadConfig (task type +
prompt/output distributions) and an ArrivalRate (rps) separately, then call
build_trace() which accepts the rate override at call time.
"""

import functools
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# Tokenizer used to measure real-dataset prompts/responses accurately (the
# synthetic path below uses a word-count heuristic instead, see _make_prompt).
# This is only a fallback: main.py calls set_tokenizer_model(args.model) before
# building workloads so lengths are counted with the *served* model's tokenizer.
# Counting a DeepSeek/Qwen prompt with a Llama tokenizer under-estimates its real
# length, so the max_tokens clamp in runner.py lets sequences overshoot
# max_model_len and trip vLLM 0.6.x's `assert num_new_tokens > 0`, killing the
# engine for the rest of the run.
TOKENIZER_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
_tokenizer_model_id = TOKENIZER_MODEL_ID


def set_tokenizer_model(model_id: str) -> None:
    """Point token counting at the served model's tokenizer (call before build).

    Real-dataset prompt/response lengths must be measured with the same tokenizer
    the vLLM server uses, otherwise the runner's context clamp is computed against
    the wrong vocabulary and sequences can exceed max_model_len at runtime.
    """
    global _tokenizer_model_id
    if model_id and model_id != _tokenizer_model_id:
        _tokenizer_model_id = model_id
        # Drop the cached tokenizer and any dataset pairs whose token lengths
        # were measured with the previous tokenizer.
        _get_tokenizer.cache_clear()
        for loader in (
            _load_sharegpt_pairs,
            _load_cnn_dailymail_pairs,
            _load_openmath_reasoning_pairs,
            _load_mbpp_pairs,
        ):
            loader.cache_clear()


class TaskType(str, Enum):
    CHAT          = "chat"
    CODE          = "code"
    SUMMARIZATION = "summarization"
    REASONING     = "reasoning"


class ArrivalPattern(str, Enum):
    POISSON = "poisson"
    BURSTY  = "bursty"


class PromptSource(str, Enum):
    SYNTHETIC = "synthetic"   # repeated-filler templates, see _make_prompt
    SHAREGPT  = "sharegpt"    # real human/GPT conversations
    CNN_DAILYMAIL = "cnn_dailymail"  # real news articles + reference summaries
    OPENMATH_REASONING = "openmath_reasoning"  # real math problems + long generated CoT solutions
    MBPP      = "mbpp"        # real Python programming problems + reference solutions
    MIXED_REAL = "mixed_real"  # interleaves all four real sources above


# ── Named arrival rates ───────────────────────────────────────────────────────

ARRIVAL_RATES: dict[str, float] = {
    "low":    2.0,   # 2 rps
    "med":    4.0,   # 4 rps
    "high":   8.0,   # 8 rps
    "vhigh":  16.0,  # 16 rps
}

# Arrival pattern is a separate, orthogonal knob from rps rate (see
# ArrivalPattern) — any rps rate above can be replayed as either Poisson or
# Bursty arrivals.
BURSTY_LABEL = "bursty"


@dataclass
class RequestSpec:
    """A single benchmark request specification."""
    request_id: str
    task_type: TaskType
    prompt: str
    prompt_token_len: int
    target_output_tokens: int
    arrival_time: float
    priority: int = 1


@dataclass
class WorkloadConfig:
    """Shape of a workload: task type, prompt/output distributions, and defaults."""
    name: str
    task_type: TaskType
    prompt_lengths: list[int]
    output_lengths: list[int]
    num_requests: int = 300
    # Default rate/pattern (used when no override is supplied to build_trace)
    default_arrival_pattern: ArrivalPattern = ArrivalPattern.POISSON
    default_target_rps: float = 2.0
    # Bursty-specific params (only used when pattern=BURSTY)
    burst_size: int = 10
    burst_interval_s: float = 5.0
    # Prompt provenance. SYNTHETIC (default) uses _make_prompt with the
    # prompt_lengths/output_lengths buckets above. Real sources instead
    # sample (prompt, target_output_tokens) pairs from a cached real
    # dataset as-is, unfiltered — prompt_lengths/output_lengths are unused
    # in that case (kept empty by convention, see the *_real entries below).
    prompt_source: PromptSource = PromptSource.SYNTHETIC
    # Optional long-tail bias for real sources: if set, only examples whose
    # prompt is at least this many tokens are eligible for sampling — the
    # bulk of most real corpora is short, so without this a fixed-size draw
    # rarely includes the (rare) long examples. Ignored for SYNTHETIC/MIXED_REAL.
    min_prompt_tokens: int = 0


# ── Prompt templates per task type ──────────────────────────────────────────

def _make_prompt(task_type: TaskType, prompt_length_tokens: int) -> str:
    base_templates = {
        TaskType.CHAT: (
            "You are a helpful assistant. The user asks: "
            "Can you explain the following concept in detail? "
            "Please provide a comprehensive answer covering all aspects. "
        ),
        TaskType.CODE: (
            "You are an expert software engineer. "
            "Write a complete, well-documented Python implementation for the following: "
            "Implement a production-grade solution with error handling, type hints, and tests. "
        ),
        TaskType.SUMMARIZATION: (
            "Summarize the following document in detail, capturing all key points, "
            "arguments, and conclusions. The document reads as follows: "
        ),
        TaskType.REASONING: (
            "Think step by step. Reason carefully through the following problem "
            "before providing your final answer. Show all intermediate steps. "
            "Problem: "
        ),
    }

    filler_by_type = {
        TaskType.CHAT: (
            "The topic involves complex interactions between various systems and components. "
            "Consider the historical context, current applications, and future implications. "
        ),
        TaskType.CODE: (
            "The system must handle concurrent requests, maintain state across sessions, "
            "validate inputs, log errors, and expose a clean public API. "
        ),
        TaskType.SUMMARIZATION: (
            "The research demonstrates significant improvements across multiple evaluation metrics. "
            "Experimental results show consistent gains over all baseline methods tested. "
        ),
        TaskType.REASONING: (
            "Given the constraints and available information, derive the optimal solution. "
            "Consider edge cases and verify your reasoning at each step. "
        ),
    }

    base = base_templates[task_type]
    filler = filler_by_type[task_type]

    target_words = int(prompt_length_tokens / 1.3)
    base_words = len(base.split())
    filler_words = len(filler.split())

    repeats = max(1, (target_words - base_words) // filler_words)
    prompt = base + (filler * repeats)
    words = prompt.split()[:target_words]
    return " ".join(words)


# ── Real-dataset prompt sources ─────────────────────────────────────────────
#
# Unlike _make_prompt (repeated filler text — trivially predictable for an
# n-gram speculative decoder, hence ~100% draft acceptance regardless of
# strategy), these load real natural-language text so token-level entropy —
# and therefore speculative-decoding acceptance rate — varies realistically
# across workloads.

@functools.lru_cache(maxsize=1)
def _get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(_tokenizer_model_id)


def _count_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text, add_special_tokens=False))


@functools.lru_cache(maxsize=1)
def _load_sharegpt_pairs() -> list[tuple[str, int]]:
    """Real human->GPT conversation turns: (human_prompt, gpt_response_token_len).

    Same corpus and first-turn convention used by vLLM's own
    benchmark_serving.py ShareGPT dataset loader.
    """
    from huggingface_hub import hf_hub_download
    import json

    logger.info("Loading ShareGPT dataset (cached after first run)...")
    path = hf_hub_download(
        repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
        filename="HTML_cleaned_raw_dataset/sg_90k_part1_html_cleaned.json",
        repo_type="dataset",
    )
    with open(path) as f:
        raw = json.load(f)

    pairs = []
    for entry in raw:
        conv = entry.get("conversations", [])
        if len(conv) < 2:
            continue
        if conv[0].get("from") != "human" or conv[1].get("from") != "gpt":
            continue
        prompt = conv[0]["value"].strip()
        response = conv[1]["value"].strip()
        if not prompt or not response:
            continue
        # Restrict to (mostly) English text: non-English turns would confound
        # the acceptance-rate-by-workload-type comparison with an
        # acceptance-rate-by-language effect.
        if not prompt.isascii() or not response.isascii():
            continue
        pairs.append((prompt, _count_tokens(response)))

    logger.info(f"Loaded {len(pairs)} ShareGPT (prompt, response_len) pairs.")
    return pairs


@functools.lru_cache(maxsize=1)
def _load_cnn_dailymail_pairs() -> list[tuple[str, int]]:
    """Real news articles + human reference summaries: (article, summary_token_len)."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    logger.info("Loading CNN/DailyMail dataset (cached after first run)...")
    path = hf_hub_download(
        repo_id="abisee/cnn_dailymail",
        filename="3.0.0/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)

    prefix = (
        "Summarize the following document in detail, capturing all key points, "
        "arguments, and conclusions. The document reads as follows: "
    )
    pairs = []
    for article, highlights in zip(df["article"], df["highlights"]):
        article = article.strip()
        highlights = highlights.strip()
        if not article or not highlights:
            continue
        pairs.append((prefix + article, _count_tokens(highlights)))

    logger.info(f"Loaded {len(pairs)} CNN/DailyMail (article, summary_len) pairs.")
    return pairs


@functools.lru_cache(maxsize=1)
def _load_openmath_reasoning_pairs() -> list[tuple[str, int]]:
    """Real math problems + long model-generated chain-of-thought solutions:
    (prompt, solution_token_len).

    Unlike GSM8K (grade-school word problems, solutions rarely exceed ~300
    tokens), OpenMathReasoning's generated_solution traces are genuinely
    long-form (median ~6.5k tokens, ~73% exceed 4k) — needed for long-context
    reasoning workloads. One shard (~22k rows) is plenty of sampling volume,
    so only the first of 144 shards is downloaded.
    """
    from huggingface_hub import hf_hub_download
    import pandas as pd

    logger.info("Loading OpenMathReasoning dataset (cached after first run)...")
    path = hf_hub_download(
        repo_id="nvidia/OpenMathReasoning",
        filename="data/cot-00000-of-00144.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)

    prefix = (
        "Think step by step. Reason carefully through the following problem "
        "before providing your final answer. Show all intermediate steps. "
        "Problem: "
    )
    pairs = []
    for problem, solution in zip(df["problem"], df["generated_solution"]):
        problem = problem.strip()
        solution = solution.strip()
        if not problem or not solution:
            continue
        pairs.append((prefix + problem, _count_tokens(solution)))

    logger.info(f"Loaded {len(pairs)} OpenMathReasoning (problem, solution_len) pairs.")
    return pairs


@functools.lru_cache(maxsize=1)
def _load_mbpp_pairs() -> list[tuple[str, int]]:
    """Real Python programming problems + reference solutions: (prompt, code_token_len).

    Pools the train/test/validation splits (skips the 10-row few-shot
    "prompt" split) for a larger sampling pool.
    """
    from huggingface_hub import hf_hub_download
    import pandas as pd

    logger.info("Loading MBPP dataset (cached after first run)...")
    prefix = (
        "You are an expert software engineer. "
        "Write a complete, well-documented Python implementation for the following: "
        "Implement a production-grade solution with error handling, type hints, and tests. "
        "Task: "
    )
    pairs = []
    for split in ("train", "test", "validation"):
        path = hf_hub_download(
            repo_id="google-research-datasets/mbpp",
            filename=f"full/{split}-00000-of-00001.parquet",
            repo_type="dataset",
        )
        df = pd.read_parquet(path)
        for text, code in zip(df["text"], df["code"]):
            text = text.strip()
            code = code.strip()
            if not text or not code:
                continue
            pairs.append((prefix + text, _count_tokens(code)))

    logger.info(f"Loaded {len(pairs)} MBPP (task, code_len) pairs.")
    return pairs


_REAL_DATASET_LOADERS = {
    PromptSource.SHAREGPT: _load_sharegpt_pairs,
    PromptSource.CNN_DAILYMAIL: _load_cnn_dailymail_pairs,
    PromptSource.OPENMATH_REASONING: _load_openmath_reasoning_pairs,
    PromptSource.MBPP: _load_mbpp_pairs,
}

# Real task type each single-source loader above actually produces — used to
# tag requests correctly when MIXED_REAL interleaves all four.
_REAL_SOURCE_TASK_TYPE = {
    PromptSource.SHAREGPT: TaskType.CHAT,
    PromptSource.CNN_DAILYMAIL: TaskType.SUMMARIZATION,
    PromptSource.OPENMATH_REASONING: TaskType.REASONING,
    PromptSource.MBPP: TaskType.CODE,
}


def _sample_real_trace(
    config: "WorkloadConfig",
    arrival_times: list[float],
    seed: int,
) -> list[RequestSpec]:
    """Build a trace from real dataset(s), used as-is (no length filtering).

    MIXED_REAL draws an even quota per task type from each of the four
    single-source datasets (regardless of how large each source corpus is —
    otherwise ShareGPT/CNN-DailyMail would drown out OpenMathReasoning/MBPP),
    then interleaves them in arrival order, tagging each request with its
    real originating task type rather than a single task_type for the whole
    trace.
    """
    rng = random.Random(seed)

    if config.prompt_source == PromptSource.MIXED_REAL:
        sources = list(_REAL_DATASET_LOADERS.items())
        quota = config.num_requests // len(sources)
        tagged_pairs: list[tuple[str, int, TaskType]] = []
        for source, loader in sources:
            task_type = _REAL_SOURCE_TASK_TYPE[source]
            pairs = loader()
            shuffled_source = pairs[:]
            rng.shuffle(shuffled_source)
            k = min(quota, len(shuffled_source))
            if k < quota:
                logger.warning(
                    f"Workload '{config.name}': {source.value} dataset only has "
                    f"{len(shuffled_source)} examples, fewer than the per-source quota {quota}."
                )
            tagged_pairs.extend((p, o, task_type) for p, o in shuffled_source[:k])
    else:
        task_type = config.task_type
        pairs = _REAL_DATASET_LOADERS[config.prompt_source]()
        if config.min_prompt_tokens > 0:
            pairs = [(p, o) for p, o in pairs if _count_tokens(p) >= config.min_prompt_tokens]
            if not pairs:
                raise ValueError(
                    f"Workload '{config.name}': no examples with prompt >= "
                    f"{config.min_prompt_tokens} tokens in {config.prompt_source.value}."
                )
        tagged_pairs = [(p, o, task_type) for p, o in pairs]
        if len(tagged_pairs) < config.num_requests:
            logger.warning(
                f"Workload '{config.name}': dataset only has {len(tagged_pairs)} examples "
                f"{'>= min_prompt_tokens ' if config.min_prompt_tokens else ''}"
                f"fewer than the requested {config.num_requests}."
            )

    n = min(config.num_requests, len(tagged_pairs))
    rng.shuffle(tagged_pairs)

    requests = []
    for prompt, response_len, task_type in tagged_pairs[:n]:
        requests.append(RequestSpec(
            request_id=f"{config.name}_{len(requests):04d}",
            task_type=task_type,
            prompt=prompt,
            prompt_token_len=_count_tokens(prompt),
            target_output_tokens=response_len,
            arrival_time=arrival_times[len(requests)],
        ))

    return requests


# ── Arrival time generators ──────────────────────────────────────────────────

def _poisson_arrivals(n: int, rps: float, seed: int = 42) -> list[float]:
    rng = random.Random(seed)
    times = []
    t = 0.0
    for _ in range(n):
        t += rng.expovariate(rps)
        times.append(t)
    return times


def _bursty_arrivals(
    n: int,
    burst_size: int,
    burst_interval_s: float,
    within_burst_rps: float = 20.0,
    seed: int = 42,
) -> list[float]:
    rng = random.Random(seed)
    times = []
    burst_start = 0.0
    within_burst_interval = 1.0 / within_burst_rps

    while len(times) < n:
        for i in range(burst_size):
            if len(times) >= n:
                break
            times.append(burst_start + i * within_burst_interval)
        burst_start += burst_interval_s

    return sorted(times[:n])


# ── Workload builder ─────────────────────────────────────────────────────────

def build_trace(
    config: WorkloadConfig,
    seed: int = 42,
    arrival_pattern: Optional[ArrivalPattern] = None,
    target_rps: Optional[float] = None,
) -> list[RequestSpec]:
    """
    Generate a request trace from a WorkloadConfig.

    arrival_pattern and target_rps override the workload's defaults when supplied.
    For BURSTY pattern, target_rps is ignored (burst params come from config).
    """
    pattern = arrival_pattern if arrival_pattern is not None else config.default_arrival_pattern
    rps     = target_rps     if target_rps     is not None else config.default_target_rps
    n       = config.num_requests

    if pattern == ArrivalPattern.BURSTY:
        arrival_times = _bursty_arrivals(
            n, config.burst_size, config.burst_interval_s, seed=seed
        )
    else:
        arrival_times = _poisson_arrivals(n, rps, seed=seed)

    if config.prompt_source != PromptSource.SYNTHETIC:
        return _sample_real_trace(config, arrival_times, seed=seed)

    length_pairs = [
        (p, o)
        for p in config.prompt_lengths
        for o in config.output_lengths
    ]

    requests = []
    for i in range(n):
        prompt_len, output_len = length_pairs[i % len(length_pairs)]
        prompt = _make_prompt(config.task_type, prompt_len)
        requests.append(RequestSpec(
            request_id=f"{config.name}_{i:04d}",
            task_type=config.task_type,
            prompt=prompt,
            prompt_token_len=prompt_len,
            target_output_tokens=output_len,
            arrival_time=arrival_times[i],
        ))

    return requests


# ── Predefined workload shapes ───────────────────────────────────────────────
# Rate is NOT embedded here — it is selected separately at run time.

WORKLOADS: dict[str, WorkloadConfig] = {

    "chat": WorkloadConfig(
        name="chat",
        task_type=TaskType.CHAT,
        prompt_lengths=[128, 256, 512, 1024],
        output_lengths=[64, 128, 256],
        num_requests=300,
        default_target_rps=2.0,
    ),
    "code": WorkloadConfig(
        name="code",
        task_type=TaskType.CODE,
        prompt_lengths=[256, 512, 1024, 2048],
        output_lengths=[256, 512, 1024],
        num_requests=200,
        default_target_rps=2.0,
    ),
    "summarization": WorkloadConfig(
        name="summarization",
        task_type=TaskType.SUMMARIZATION,
        prompt_lengths=[2048, 4096, 8192],
        output_lengths=[128, 256, 512],
        num_requests=150,
        default_target_rps=2.0,
    ),
    "reasoning": WorkloadConfig(
        name="reasoning",
        task_type=TaskType.REASONING,
        prompt_lengths=[256, 512, 1024],
        output_lengths=[512, 1024, 2048],
        num_requests=200,
        default_arrival_pattern=ArrivalPattern.BURSTY,
        burst_size=8,
        burst_interval_s=10.0,
    ),
    "mixed": WorkloadConfig(
        name="mixed",
        task_type=TaskType.CHAT,
        prompt_lengths=[128, 512, 1024, 4096],
        output_lengths=[64, 256, 512, 1024],
        num_requests=400,
        default_target_rps=2.0,
    ),

    # ── Real-data workloads ──────────────────────────────────────────────
    # Prompts and target output lengths are taken as-is from the real
    # corpus (no length filtering/clamping) — prompt_lengths/output_lengths
    # are unused for these (see prompt_source). Use these to measure
    # realistic speculative-decoding draft acceptance rates.
    "chat_real": WorkloadConfig(
        name="chat_real",
        task_type=TaskType.CHAT,
        prompt_lengths=[],
        output_lengths=[],
        num_requests=300,
        default_target_rps=2.0,
        prompt_source=PromptSource.SHAREGPT,
    ),
    # Long-tail variant: only ~1.2% of ShareGPT prompts exceed 1000 tokens, so
    # a plain random draw of chat_real almost never samples them. This biases
    # sampling to that long tail (only 679 examples clear 500 tokens, so
    # num_requests is capped accordingly).
    "chat_real_long": WorkloadConfig(
        name="chat_real_long",
        task_type=TaskType.CHAT,
        prompt_lengths=[],
        output_lengths=[],
        num_requests=300,
        default_target_rps=2.0,
        prompt_source=PromptSource.SHAREGPT,
        min_prompt_tokens=500,
    ),
    "summarization_real": WorkloadConfig(
        name="summarization_real",
        task_type=TaskType.SUMMARIZATION,
        prompt_lengths=[],
        output_lengths=[],
        num_requests=150,
        default_target_rps=2.0,
        prompt_source=PromptSource.CNN_DAILYMAIL,
    ),
    # Long-tail variant: biases toward CNN/DailyMail's longer articles
    # (~3700 examples clear 1000 tokens) for longer-context summarization runs.
    "summarization_real_long": WorkloadConfig(
        name="summarization_real_long",
        task_type=TaskType.SUMMARIZATION,
        prompt_lengths=[],
        output_lengths=[],
        num_requests=150,
        default_target_rps=2.0,
        prompt_source=PromptSource.CNN_DAILYMAIL,
        min_prompt_tokens=1000,
    ),
    "reasoning_real": WorkloadConfig(
        name="reasoning_real",
        task_type=TaskType.REASONING,
        prompt_lengths=[],
        output_lengths=[],
        num_requests=200,
        default_arrival_pattern=ArrivalPattern.BURSTY,
        burst_size=8,
        burst_interval_s=10.0,
        prompt_source=PromptSource.OPENMATH_REASONING,
    ),
    "code_real": WorkloadConfig(
        name="code_real",
        task_type=TaskType.CODE,
        prompt_lengths=[],
        output_lengths=[],
        num_requests=200,
        default_target_rps=2.0,
        prompt_source=PromptSource.MBPP,
    ),
    "mixed_real": WorkloadConfig(
        name="mixed_real",
        # task_type is unused for MIXED_REAL — each request is tagged with
        # its real originating task type (chat/summarization/reasoning/code)
        # by _sample_real_trace, not this single default.
        task_type=TaskType.CHAT,
        prompt_lengths=[],
        output_lengths=[],
        num_requests=400,
        default_target_rps=2.0,
        prompt_source=PromptSource.MIXED_REAL,
    ),
}


def get_workload(name: str) -> WorkloadConfig:
    if name not in WORKLOADS:
        raise ValueError(f"Unknown workload '{name}'. Available: {list(WORKLOADS.keys())}")
    return WORKLOADS[name]
