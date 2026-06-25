"""
InferenceBench TUI — interactive launcher.

Usage:
    python tui.py
"""

import subprocess
import sys
import os
from datetime import datetime

import questionary
from questionary import Style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

from benchmark.launcher import kill_stale_gpu_workers
from config.strategies import STRATEGIES, Strategy, get_strategies_for_dimension
from config.workloads import ARRIVAL_RATES, ArrivalPattern, BURSTY_LABEL, WORKLOADS

console = Console()


def detect_gpus() -> list[dict]:
    """Query nvidia-smi for available GPUs. Returns [] if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue
            idx, name, mem_total, mem_free = parts
            gpus.append({
                "index": idx,
                "name": name,
                "mem_total_gib": float(mem_total) / 1024,
                "mem_free_gib": float(mem_free) / 1024,
            })
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

STYLE = Style([
    ("qmark",        "fg:#00bfff bold"),
    ("question",     "bold"),
    ("answer",       "fg:#00ff99 bold"),
    ("pointer",      "fg:#00bfff bold"),
    ("highlighted",  "fg:#00bfff bold"),
    ("selected",     "fg:#00ff99"),
    ("separator",    "fg:#444444"),
    ("instruction",  "fg:#888888"),
])

DIMENSION_GROUPS = {
    "chunked_prefill":  ["baseline_full_fp16_tp1", "chunked_256_fp16_tp1", "chunked_512_fp16_tp1", "chunked_1024_fp16_tp1"],
    "tensor_parallel":  ["baseline_full_fp16_tp1", "baseline_full_fp16_tp2", "baseline_full_fp16_tp4", "chunked_512_fp16_tp2"],
    "speculation":      ["baseline_full_fp16_tp1", "spec_ngram3_fp16_tp1", "spec_ngram5_fp16_tp1"],
    "quantization":     ["baseline_full_fp16_tp1", "baseline_full_fp8_tp1", "chunked_512_fp8_tp1"],
    "interaction":      ["baseline_full_fp16_tp1", "chunked_512_fp16_tp1", "spec_ngram3_fp16_tp1", "chunked_512_spec_ngram3_fp16_tp1"],
}


def print_header():
    console.print(Panel.fit(
        Text("InferenceBench", style="bold cyan") +
        Text(" — LLM Execution Strategy Benchmark", style="dim"),
        border_style="cyan",
    ))
    console.print()


def strategy_table(selected: list[str]):
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", pad_edge=False)
    t.add_column("Strategy", style="white")
    t.add_column("TP", justify="center")
    t.add_column("MaxSeqs", justify="center")
    t.add_column("Speculation", justify="center")
    t.add_column("Quant", justify="center")
    for name in selected:
        s = STRATEGIES[name]
        t.add_row(
            name,
            str(s.tensor_parallel_size),
            str(s.max_num_seqs) if s.max_num_seqs is not None else "default",
            f"✓ ngram={s.ngram_prompt_lookup_max}" if s.enable_speculation else "—",
            s.quantization or "—",
        )
    console.print(t)


PRESET_MODELS = {
    "Llama": [
        ("Llama-3.1-8B-Instruct",  "meta-llama/Llama-3.1-8B-Instruct"),
        ("Llama-3.1-70B-Instruct", "meta-llama/Llama-3.1-70B-Instruct"),
        ("Llama-3.2-3B-Instruct",  "meta-llama/Llama-3.2-3B-Instruct"),
        ("Llama-3.2-11B-Vision",   "meta-llama/Llama-3.2-11B-Vision-Instruct"),
        ("Llama-3.3-70B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"),
    ],
    "Mistral": [
        ("Mistral-7B-Instruct-v0.3",  "mistralai/Mistral-7B-Instruct-v0.3"),
        ("Mistral-Nemo-12B",          "mistralai/Mistral-Nemo-Instruct-2407"),
        ("Mixtral-8x7B-Instruct",     "mistralai/Mixtral-8x7B-Instruct-v0.1"),
        ("Mistral-Small-3.1-24B",     "mistralai/Mistral-Small-3.1-24B-Instruct-2503"),
    ],
    "DeepSeek": [
        ("DeepSeek-R1-Distill-Qwen-7B",  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        ("DeepSeek-R1-Distill-Llama-8B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
        ("DeepSeek-V2-Lite-Chat",        "deepseek-ai/DeepSeek-V2-Lite-Chat"),
        ("DeepSeek-Coder-V2-Lite",       "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"),
    ],
    "Qwen": [
        ("Qwen2.5-7B-Instruct",   "Qwen/Qwen2.5-7B-Instruct"),
        ("Qwen2.5-14B-Instruct",  "Qwen/Qwen2.5-14B-Instruct"),
        ("Qwen2.5-Coder-7B",      "Qwen/Qwen2.5-Coder-7B-Instruct"),
        ("QwQ-32B",               "Qwen/QwQ-32B"),
    ],
    "Gemma": [
        ("Gemma-2-9B-it",   "google/gemma-2-9b-it"),
        ("Gemma-2-27B-it",  "google/gemma-2-27b-it"),
        ("Gemma-3-12B-it",  "google/gemma-3-12b-it"),
    ],
    "Phi": [
        ("Phi-3.5-mini-instruct",  "microsoft/Phi-3.5-mini-instruct"),
        ("Phi-4",                  "microsoft/phi-4"),
        ("Phi-4-mini-instruct",    "microsoft/Phi-4-mini-instruct"),
    ],
}

CUSTOM_OPTION  = "— Enter custom model ID —"
BACK_OPTION    = "← Back to families"


def ask_model() -> str:
    # Step 1: pick family or custom
    family_choices = list(PRESET_MODELS.keys()) + [CUSTOM_OPTION]
    family = questionary.select(
        "Select model family:",
        choices=family_choices,
        style=STYLE,
    ).ask()

    if family is None:
        return ""

    if family == CUSTOM_OPTION:
        return questionary.text(
            "Paste HuggingFace model ID or local path:",
            style=STYLE,
        ).ask() or ""

    # Step 2: pick specific model within family
    presets = PRESET_MODELS[family]
    model_choices = [
        questionary.Choice(label, value=hf_id)
        for label, hf_id in presets
    ] + [questionary.Choice(CUSTOM_OPTION, value=CUSTOM_OPTION),
         questionary.Choice(BACK_OPTION,   value=BACK_OPTION)]

    selected = questionary.select(
        f"Select {family} model:",
        choices=model_choices,
        style=STYLE,
    ).ask()

    if selected == BACK_OPTION:
        return ask_model()  # go back to family picker

    if selected == CUSTOM_OPTION:
        return questionary.text(
            "Paste HuggingFace model ID or local path:",
            style=STYLE,
        ).ask() or ""

    return selected or ""


def ask_strategies() -> list[str]:
    mode = questionary.select(
        "How do you want to select strategies?",
        choices=[
            questionary.Choice("Pick individual strategies",  value="manual"),
            questionary.Choice("By dimension (ablation)",     value="dimension"),
            questionary.Choice("Run all strategies",          value="all"),
        ],
        style=STYLE,
    ).ask()

    if mode == "all":
        return list(STRATEGIES.keys())

    if mode == "dimension":
        dim = questionary.select(
            "Select dimension to ablate:",
            choices=list(DIMENSION_GROUPS.keys()),
            style=STYLE,
        ).ask()
        return DIMENSION_GROUPS[dim]

    # manual
    choices = [
        questionary.Choice(name, value=name, checked=False)
        for name in STRATEGIES
    ]
    selected = questionary.checkbox(
        "Select strategies  [Space=toggle, Enter=confirm]:",
        choices=choices,
        style=STYLE,
    ).ask()
    return selected or []


def ask_workloads() -> tuple[list[str], list[str]]:
    """Return (workload_names, rate_labels)."""

    # ── Step 1: workload shapes ──────────────────────────────────────────────
    wl_choices = [
        questionary.Choice(
            f"{name}  [n={WORKLOADS[name].num_requests}, "
            f"{WORKLOADS[name].task_type.value}]",
            value=name,
            checked=False,
        )
        for name in WORKLOADS
    ]
    selected_wl = questionary.checkbox(
        "Select workload(s)  [Space=toggle, Enter=confirm]:",
        choices=wl_choices,
        style=STYLE,
    ).ask() or []

    # ── Step 2: arrival rates ────────────────────────────────────────────────
    rate_choices = [
        questionary.Choice("low    — 2 rps  (Poisson)",  value="low",    checked=False),
        questionary.Choice("med    — 4 rps  (Poisson)",  value="med",    checked=False),
        questionary.Choice("high   — 8 rps  (Poisson)",  value="high",   checked=False),
        questionary.Choice("bursty — burst pattern",     value="bursty", checked=False),
    ]
    selected_rates = questionary.checkbox(
        "Select arrival rate(s)  [Space=toggle, Enter=confirm]:",
        choices=rate_choices,
        style=STYLE,
    ).ask() or []

    return selected_wl, selected_rates


MAX_NUM_SEQS_OPTIONS = [
    ("default", None),
    ("16  — low parallelism",   16),
    ("32  — medium",            32),
    ("64  — high parallelism",  64),
    ("128 — very high",        128),
]

def ask_max_num_seqs() -> list[int | None]:
    """
    Multi-select for scheduler concurrency (max_num_seqs).
    Each selected value produces a separate run per strategy.
    Returns list of values (None = use strategy default).
    """
    choices = [
        questionary.Choice(label, value=val, checked=False)
        for label, val in MAX_NUM_SEQS_OPTIONS
    ]
    selected = questionary.checkbox(
        "Select max_num_seqs (scheduler concurrency)  [Space=toggle, Enter=confirm]:\n"
        "  Each value runs as a separate sweep — pick one to keep it simple:",
        choices=choices,
        style=STYLE,
    ).ask() or []
    return selected if selected else [None]   # default if none picked


def ask_gpus() -> str:
    """Let the user pick from detected GPUs, falling back to free-text entry."""
    gpus = detect_gpus()
    if not gpus:
        return questionary.text("GPU indices (comma-separated):", default="0", style=STYLE).ask()

    choices = [
        questionary.Choice(
            f"GPU {g['index']}: {g['name']}  ({g['mem_free_gib']:.1f}/{g['mem_total_gib']:.1f} GiB free)",
            value=g["index"],
            checked=(g["index"] == "0"),
        )
        for g in gpus
    ]
    selected = questionary.checkbox(
        "Select GPU(s)  [Space=toggle, Enter=confirm]:",
        choices=choices,
        style=STYLE,
    ).ask()
    if not selected:
        console.print("[yellow]No GPU selected — defaulting to GPU 0.[/yellow]")
        return "0"
    return ",".join(selected)


def ask_options() -> dict:
    gpu = ask_gpus()
    port = questionary.text("Base port for vLLM:", default="8100", style=STYLE).ask()
    timeout = questionary.text("Request timeout (seconds):", default="300", style=STYLE).ask()

    num_requests = questionary.select(
        "Number of requests per workload:",
        choices=[
            questionary.Choice("50   — quick smoke test (~1–2 min)",   value="50"),
            questionary.Choice("100  — fast run (~3–5 min)",           value="100"),
            questionary.Choice("200  — standard (~8–10 min)",          value="200"),
            questionary.Choice("300  — default (~12–15 min)",          value="300"),
            questionary.Choice("500  — thorough (~20–25 min)",         value="500"),
            questionary.Choice("1000 — full run  (~40–50 min)",        value="1000"),
            questionary.Choice("Custom",                                value="custom"),
        ],
        style=STYLE,
    ).ask()

    if num_requests == "custom":
        num_requests = questionary.text(
            "Enter number of requests:", default="300", style=STYLE
        ).ask()

    return {"gpu": gpu, "port": port, "timeout": timeout,
            "num_requests": num_requests, "max_concurrent": "64"}


def confirm_and_run(
    model: str,
    strategies: list[str],
    workloads: list[str],
    rates: list[str],
    seqs_values: list[int | None],
    opts: dict,
):
    console.print()
    console.print("[bold cyan]Run summary[/bold cyan]")
    console.print(f"  Model:      [yellow]{model}[/yellow]")
    for wl in workloads:
        wl_obj = WORKLOADS[wl]
        console.print(f"  Workload:   [yellow]{wl}[/yellow]  [dim](n={wl_obj.num_requests}, {wl_obj.task_type.value})[/dim]")
    console.print(f"  Rates:      [yellow]{', '.join(rates)}[/yellow]")
    console.print(f"  max_num_seqs: [yellow]{', '.join(str(v) if v else 'default' for v in seqs_values)}[/yellow]")
    console.print(f"  GPU(s):     [yellow]{opts['gpu']}[/yellow]")
    console.print()
    strategy_table(strategies)

    if not questionary.confirm("Launch benchmark?", default=True, style=STYLE).ask():
        console.print("[dim]Cancelled.[/dim]")
        return

    kill_stale_gpu_workers()

    python = sys.executable
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build the list of (strategy_name, max_num_seqs_override) combos.
    # If only one seqs value and it's None (default), no override is applied.
    combos: list[tuple[str, int | None]] = [
        (strat, seqs)
        for strat in strategies
        for seqs in seqs_values
    ]

    base_cmd = [
        python, "main.py",
        "--model", model,
        "--run-id", run_id,
        "--gpu-indices", opts["gpu"],
        "--port", opts["port"],
        "--request-timeout", opts["timeout"],
        "--num-requests", opts["num_requests"],
        "--max-concurrent", opts["max_concurrent"],
        "--workloads",
    ] + workloads + ["--rates"] + rates

    console.print(f"[dim]Run ID: {run_id} (shared across all runs)[/dim]\n")
    console.print()

    n_total = len(combos)
    for i, (strat, seqs) in enumerate(combos, 1):
        label = strat if seqs is None else f"{strat} [max_num_seqs={seqs}]"
        console.rule(f"[cyan]{i}/{n_total} — {label}[/cyan]")
        cmd = base_cmd + ["--strategy", strat]
        if seqs is not None:
            cmd += ["--max-num-seqs-override", str(seqs)]
        console.print(f"[dim]Command: {' '.join(cmd)}[/dim]\n")
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            console.print(f"[red]Run {label} failed (returncode={ret.returncode}). Continuing...[/red]")

    console.print()
    console.print("[bold green]✓ Benchmark finished.[/bold green]")
    console.print(f"Results in: [cyan]results/{run_id}[/cyan]")


def main():
    print_header()

    model = ask_model()
    if not model:
        return

    strategies = ask_strategies()
    if not strategies:
        console.print("[red]No strategies selected. Exiting.[/red]")
        return

    workloads, rates = ask_workloads()
    if not workloads:
        console.print("[red]No workloads selected. Exiting.[/red]")
        return
    if not rates:
        console.print("[red]No arrival rates selected. Exiting.[/red]")
        return

    seqs_values = ask_max_num_seqs()

    opts = ask_options()
    confirm_and_run(model, strategies, workloads, rates, seqs_values, opts)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
