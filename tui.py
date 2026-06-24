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
from config.strategies import STRATEGIES, get_strategies_for_dimension
from config.workloads import WORKLOADS

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


def ask_workloads() -> list[str]:
    choices = [
        questionary.Choice(
            f"{name}  [n={WORKLOADS[name].num_requests}, {WORKLOADS[name].arrival_pattern.name.lower()}]",
            value=name,
            checked=False,
        )
        for name in WORKLOADS
    ]
    selected = questionary.checkbox(
        "Select workloads  [Space=toggle, Enter=confirm]:",
        choices=choices,
        style=STYLE,
    ).ask()
    return selected or []


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
            questionary.Choice("Custom",                                value="custom"),
        ],
        style=STYLE,
    ).ask()

    if num_requests == "custom":
        num_requests = questionary.text(
            "Enter number of requests:", default="300", style=STYLE
        ).ask()

    max_concurrent = questionary.select(
        "Max concurrent requests (lower = less vLLM backpressure):",
        choices=[
            questionary.Choice("8",          value="8"),
            questionary.Choice("16 (default)", value="16"),
            questionary.Choice("24",         value="24"),
            questionary.Choice("32",         value="32"),
            questionary.Choice("40",         value="40"),
            questionary.Choice("48",         value="48"),
            questionary.Choice("Custom",     value="custom"),
        ],
        style=STYLE,
    ).ask()

    if max_concurrent == "custom":
        max_concurrent = questionary.text(
            "Enter max concurrent requests:", default="16", style=STYLE
        ).ask()

    return {"gpu": gpu, "port": port, "timeout": timeout,
            "num_requests": num_requests, "max_concurrent": max_concurrent}


def confirm_and_run(model: str, strategies: list[str], workloads: list[str], opts: dict):
    console.print()
    console.print("[bold cyan]Run summary[/bold cyan]")
    console.print(f"  Model:      [yellow]{model}[/yellow]")
    for wl in workloads:
        wl_obj = WORKLOADS[wl]
        console.print(f"  Workload:   [yellow]{wl}[/yellow]  [dim](n={wl_obj.num_requests}, {wl_obj.arrival_pattern.name.lower()})[/dim]")
    console.print(f"  GPU(s):     [yellow]{opts['gpu']}[/yellow]")
    console.print()
    strategy_table(strategies)

    if not questionary.confirm("Launch benchmark?", default=True, style=STYLE).ask():
        console.print("[dim]Cancelled.[/dim]")
        return

    # Guard against orphaned vLLM workers left on the GPU from a previous
    # crashed/interrupted run (e.g. the TUI itself was killed mid-benchmark).
    kill_stale_gpu_workers()

    python = sys.executable
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    ] + workloads

    console.print()

    if len(strategies) == len(STRATEGIES):
        # all strategies: use --run-all (single process, single run_id)
        cmd = base_cmd + ["--run-all"]
        # remove --workloads if using --run-all without explicit workloads list
        console.print(f"[dim]Command: {' '.join(cmd)}[/dim]\n")
        subprocess.run(cmd)
    elif len(strategies) == 1:
        cmd = base_cmd + ["--strategy", strategies[0]]
        console.print(f"[dim]Command: {' '.join(cmd)}[/dim]\n")
        subprocess.run(cmd)
    else:
        # Multiple strategies: run sequentially, sharing the same run_id
        # so all results accumulate in one directory and the final report covers all
        console.print(f"[dim]Run ID: {run_id} (shared across all strategies)[/dim]\n")
        for i, strat in enumerate(strategies, 1):
            console.rule(f"[cyan]{i}/{len(strategies)} — {strat}[/cyan]")
            cmd = base_cmd + ["--strategy", strat]
            console.print(f"[dim]Command: {' '.join(cmd)}[/dim]\n")
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                console.print(f"[red]Strategy {strat} failed (returncode={ret.returncode}). Continuing...[/red]")

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

    workloads = ask_workloads()
    if not workloads:
        console.print("[red]No workloads selected. Exiting.[/red]")
        return

    opts = ask_options()
    confirm_and_run(model, strategies, workloads, opts)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
