"""
Workload configuration and request trace generation.

Covers four task types x five prompt lengths x four output lengths.
Arrival processes: Poisson (low/med/high rps), bursty.

Rate is decoupled from workload shape: pick a WorkloadConfig (task type +
prompt/output distributions) and an ArrivalRate (rps) separately, then call
build_trace() which accepts the rate override at call time.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskType(str, Enum):
    CHAT          = "chat"
    CODE          = "code"
    SUMMARIZATION = "summarization"
    REASONING     = "reasoning"


class ArrivalPattern(str, Enum):
    POISSON = "poisson"
    BURSTY  = "bursty"


# ── Named arrival rates ───────────────────────────────────────────────────────

ARRIVAL_RATES: dict[str, float] = {
    "low":    2.0,   # 2 rps
    "med":    4.0,   # 4 rps
    "high":   8.0,   # 8 rps
}
# "bursty" is a separate pattern, not a rps value.

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
        output_lengths=[512, 1024],
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
}


def get_workload(name: str) -> WorkloadConfig:
    if name not in WORKLOADS:
        raise ValueError(f"Unknown workload '{name}'. Available: {list(WORKLOADS.keys())}")
    return WORKLOADS[name]
