"""
Workload configuration and request trace generation.

Covers four task types x five prompt lengths x four output lengths.
Arrival processes: uniform, Poisson (low/high load), bursty.
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Generator


class TaskType(str, Enum):
    CHAT          = "chat"
    CODE          = "code"
    SUMMARIZATION = "summarization"
    REASONING     = "reasoning"


class ArrivalPattern(str, Enum):
    UNIFORM  = "uniform"
    POISSON_LOW  = "poisson_low"
    POISSON_HIGH = "poisson_high"
    BURSTY   = "bursty"


@dataclass
class RequestSpec:
    """A single benchmark request specification."""
    request_id: str
    task_type: TaskType
    prompt: str
    prompt_token_len: int       # approximate, measured after tokenization
    target_output_tokens: int   # max_tokens sent to vLLM
    arrival_time: float         # seconds from trace start (0.0 = immediate)
    priority: int = 1           # 1=normal, 2=high


@dataclass
class WorkloadConfig:
    name: str
    task_type: TaskType
    prompt_lengths: list[int]               # target prompt token counts
    output_lengths: list[int]               # max_tokens values
    arrival_pattern: ArrivalPattern
    num_requests: int = 200
    target_rps: float = 2.0                 # requests per second (Poisson mean)
    burst_size: int = 10                    # requests per burst (bursty pattern)
    burst_interval_s: float = 5.0          # seconds between bursts


# ── Prompt templates per task type ──────────────────────────────────────────

def _make_prompt(task_type: TaskType, prompt_length_tokens: int) -> str:
    """
    Generate a synthetic prompt of approximately prompt_length_tokens tokens.
    Uses task-appropriate templates; pads with domain-relevant filler.
    """
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

    # ~1.3 tokens per word average; approximate target
    target_words = int(prompt_length_tokens / 1.3)
    base_words = len(base.split())
    filler_words = len(filler.split())

    repeats = max(1, (target_words - base_words) // filler_words)
    prompt = base + (filler * repeats)

    # trim to approximate target
    words = prompt.split()[:target_words]
    return " ".join(words)


# ── Arrival time generators ──────────────────────────────────────────────────

def _uniform_arrivals(n: int, rps: float) -> list[float]:
    interval = 1.0 / rps
    return [i * interval for i in range(n)]


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
    t = 0.0
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

def build_trace(config: WorkloadConfig, seed: int = 42) -> list[RequestSpec]:
    """
    Generate a request trace from a WorkloadConfig.
    Cycles through prompt_lengths x output_lengths combinations.
    """
    rng = random.Random(seed)
    n = config.num_requests

    # arrival times
    if config.arrival_pattern == ArrivalPattern.UNIFORM:
        arrival_times = _uniform_arrivals(n, config.target_rps)
    elif config.arrival_pattern in (ArrivalPattern.POISSON_LOW, ArrivalPattern.POISSON_HIGH):
        arrival_times = _poisson_arrivals(n, config.target_rps, seed=seed)
    else:  # BURSTY
        arrival_times = _bursty_arrivals(
            n, config.burst_size, config.burst_interval_s, seed=seed
        )

    # build requests
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


# ── Predefined workload configurations ───────────────────────────────────────

WORKLOADS: dict[str, WorkloadConfig] = {

    "chat_poisson_low": WorkloadConfig(
        name="chat_poisson_low",
        task_type=TaskType.CHAT,
        prompt_lengths=[128, 256, 512, 1024],
        output_lengths=[64, 128, 256],
        arrival_pattern=ArrivalPattern.POISSON_LOW,
        num_requests=300,
        target_rps=1.0,
    ),
    "chat_poisson_high": WorkloadConfig(
        name="chat_poisson_high",
        task_type=TaskType.CHAT,
        prompt_lengths=[128, 256, 512, 1024],
        output_lengths=[64, 128, 256],
        arrival_pattern=ArrivalPattern.POISSON_HIGH,
        num_requests=300,
        target_rps=4.0,
    ),
    "code_uniform": WorkloadConfig(
        name="code_uniform",
        task_type=TaskType.CODE,
        prompt_lengths=[256, 512, 1024, 2048],
        output_lengths=[256, 512, 1024],
        arrival_pattern=ArrivalPattern.UNIFORM,
        num_requests=200,
        target_rps=1.5,
    ),
    "summarization_long_prompt": WorkloadConfig(
        name="summarization_long_prompt",
        task_type=TaskType.SUMMARIZATION,
        prompt_lengths=[2048, 4096, 8192],
        output_lengths=[128, 256, 512],
        arrival_pattern=ArrivalPattern.POISSON_LOW,
        num_requests=150,
        target_rps=0.5,
    ),
    "reasoning_bursty": WorkloadConfig(
        name="reasoning_bursty",
        task_type=TaskType.REASONING,
        prompt_lengths=[256, 512, 1024],
        output_lengths=[512, 1024],          # reasoning models produce long outputs
        arrival_pattern=ArrivalPattern.BURSTY,
        num_requests=200,
        target_rps=2.0,
        burst_size=8,
        burst_interval_s=10.0,
    ),
    "mixed_all": WorkloadConfig(
        name="mixed_all",
        task_type=TaskType.CHAT,             # task_type overridden per request below
        prompt_lengths=[128, 512, 1024, 4096],
        output_lengths=[64, 256, 512, 1024],
        arrival_pattern=ArrivalPattern.POISSON_LOW,
        num_requests=400,
        target_rps=2.0,
    ),
}


def get_workload(name: str) -> WorkloadConfig:
    if name not in WORKLOADS:
        raise ValueError(f"Unknown workload '{name}'. Available: {list(WORKLOADS.keys())}")
    return WORKLOADS[name]
