# LLM Execution Strategy Benchmark

> **Research artifact for:** *"Compiler-Inspired Execution Planning for LLM Request Dispatch"*  
> Target venue: ICSE 2027

---

## Research Motivation

Modern LLM serving systems (vLLM, SGLang, TensorRT-LLM) fix their execution strategy at deployment time. Every incoming request — regardless of its prompt length, expected output length, or task type — is served identically: same prefill mode, same parallelism degree, same speculation policy.

This is analogous to a database that ignores query statistics and always uses a full table scan.

**The core empirical question this benchmark answers:**

> *Does no single execution strategy dominate across all request characteristics and workload patterns? If so, what is the cost of static misconfiguration, and which request features predict the optimal strategy?*

These findings are the empirical foundation for a per-request execution planner (Paper 2), analogous to a query optimizer in databases.

---

## Project Structure

```
InferenceBench/
├── main.py                         Entry point and CLI orchestrator
├── tui.py                          Interactive terminal UI (strategy/workload picker)
├── requirements.txt
├── RESEARCH_OVERVIEW.md            Motivation, contributions, paper outline
├── config/
│   ├── __init__.py
│   ├── strategies.py               Strategy definitions + vLLM CLI arg builder
│   └── workloads.py                Workload configs + Poisson/bursty trace generator
└── benchmark/
    ├── __init__.py
    ├── launcher.py                 vLLM subprocess lifecycle (start, health-poll, teardown)
    ├── runner.py                   Async trace replayer with SSE streaming + TTFT measurement
    ├── gpu_monitor.py              Background GPU sampler (pynvml / nvidia-smi fallback)
    ├── storage.py                  CSV + JSON persistence per run
    └── analysis/
        ├── __init__.py
        ├── cost_model.py           Separability analysis, dominance pivot, GBT cost model fitting
        └── report.py               Paper-quality figures (PDF) + LaTeX tables
```

**Important:** the three `__init__.py` files (in `benchmark/`, `benchmark/analysis/`, and `config/`) are required for Python's package import resolution. They are intentionally empty — do not delete them.

---

## Execution Strategy Space

| Dimension | Options |
|---|---|
| Prefill mode | Full prefill, Chunked-256, Chunked-512, Chunked-1024 |
| Tensor parallelism | TP-1, TP-2, TP-4 |
| Speculative decoding | None, ngram-3, ngram-5 |
| Quantization | fp16, fp8 |
| Combinations | Chunked + speculation interaction |

12 strategies defined (fractional factorial design).

## Workload Space

| Workload | Task type | Prompt lengths | Output lengths | Arrival pattern |
|---|---|---|---|---|
| `chat_poisson_low` | Chat | 128–1024 | 64–256 | Poisson λ=1.0 rps |
| `chat_poisson_high` | Chat | 128–1024 | 64–256 | Poisson λ=4.0 rps |
| `code_uniform` | Code generation | 256–2048 | 256–1024 | Uniform 1.5 rps |
| `summarization_long_prompt` | Summarization | 2048–8192 | 128–512 | Poisson λ=0.5 rps |
| `reasoning_bursty` | CoT reasoning | 256–1024 | 512–1024 | Bursty (8 req/burst) |
| `mixed_all` | Mixed | 128–4096 | 64–1024 | Poisson λ=2.0 rps |

## Metrics Collected Per Request

- **TTFT** (Time to First Token) — via SSE streaming, millisecond resolution
- **E2E latency** — full request completion time
- **Tokens per second** — decode throughput
- **KV cache utilization** — from vLLM Prometheus `/metrics`
- **Preemption count** — number of KV evictions during the run
- **GPU utilization, memory, power** — 1 Hz background sampling via pynvml

---

## Hardware Requirements

| Configuration | Supported strategies |
|---|---|
| 1 × GPU (≥16 GB VRAM) | All TP-1 strategies (10 of 12) |
| 2 × GPU | All TP-1 + TP-2 strategies (11 of 12) |
| 4 × GPU | Full strategy space including TP-4 |

The launcher automatically skips strategies that require more GPUs than available.

**VRAM guidance for Llama-3.1-8B at fp16, max context 8192:**
- V100-32GB: supported (validated)
- A100-40GB+: recommended for full strategy space

---

## Installation

```bash
# 1. Enter project root
cd InferenceBench

# 2. Create and activate a virtualenv
python -m venv .venv
source .venv/bin/activate

# 3. Install benchmark dependencies
pip install -r requirements.txt

# 4. Install PyTorch matching your CUDA driver
#    CUDA 12.1 / 12.2 (e.g. V100, A100 with driver 535.x):
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

#    CUDA 12.4+ (e.g. H100, A100 with driver 550+):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 5. Install vLLM and pin compatible transformers
pip install vllm==0.6.0
pip install "transformers==4.45.2" "tokenizers==0.20.3"

# 6. Install TUI dependencies
pip install questionary rich

# 7. Log in to HuggingFace (required for gated models like Llama)
huggingface-cli login

# 8. Verify setup
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
python -c "import vllm; print(vllm.__version__)"
python -c "from benchmark.runner import run_benchmark; print('imports OK')"
```

---

## Running the Benchmark

**All commands must be run from the project root** (the directory containing `main.py`).

### Interactive TUI — recommended for most use cases

```bash
python tui.py
```

Walks you through selecting model, strategies, workloads, and GPU options with arrow-key menus, then launches the benchmark.

### CLI — single strategy × workload (sanity check, ~5 min)

```bash
python main.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --strategy baseline_full_fp16_tp1 \
  --workload chat_poisson_low
```

### CLI — ablation over one dimension (~1–2 hours per dimension)

```bash
# Prefill mode ablation
python main.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dimension chunked_prefill \
  --workload chat_poisson_high

# Speculation ablation
python main.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dimension speculation \
  --workload reasoning_bursty

# Available dimensions: chunked_prefill | tensor_parallel | speculation | quantization | interaction
```

### CLI — multiple workloads in one run

```bash
python main.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dimension chunked_prefill \
  --workloads chat_poisson_low chat_poisson_high summarization_long_prompt
```

### CLI — full benchmark (all strategies × all workloads, ~8–12 hours)

```bash
python main.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --run-all \
  --gpu-indices 0         # single GPU
  # --gpu-indices 0,1     # two GPUs
```

### Regenerate report from existing results

```bash
python main.py \
  --model placeholder \
  --report-only \
  --run-id 20260615_154412
```

---

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--model` | required | HuggingFace model ID or local path |
| `--results-dir` | `results/` | Root directory for all output |
| `--run-id` | timestamp | Identifier for this run |
| `--gpu-indices` | `0` | Comma-separated GPU indices |
| `--port` | `8100` | vLLM server port |
| `--max-concurrent` | `32` | Max in-flight requests |
| `--request-timeout` | `300` | Per-request timeout (seconds) |
| `--seed` | `42` | Reproducibility seed |
| `--no-gpu-monitor` | off | Disable GPU sampling |
| `--extra-vllm-args` | — | Additional args forwarded to vLLM |

**Execution mode (mutually exclusive, one required):**

| Flag | Description |
|---|---|
| `--run-all` | All 12 strategies × all 6 workloads |
| `--strategy <name>` | Single named strategy |
| `--dimension <name>` | Ablation group for one dimension |
| `--report-only` | Report generation only (requires `--run-id`) |

---

## Output Structure

```
results/
└── {run_id}/
    ├── requests.csv              Per-request metrics (TTFT, E2E, TPS, KV cache %, preemptions)
    ├── gpu_samples.csv           1 Hz GPU utilization / memory / power samples
    ├── summary.json              P50/P95/P99 aggregates per strategy × workload
    ├── vllm_{strategy}.log       vLLM server logs per strategy
    ├── figures/
    │   ├── ttft_cdf.pdf          TTFT CDF by strategy
    │   ├── dominance_heatmap.pdf Best strategy per (prompt_len, output_len) regime
    │   ├── ttft_boxplot.pdf      TTFT distribution comparison
    │   ├── e2e_boxplot.pdf       E2E distribution comparison
    │   ├── kv_cache_vs_ttft.pdf  KV pressure vs. TTFT scatter
    │   └── tps_by_output_length.pdf  Decode throughput vs. output length
    ├── tables/
    │   ├── summary.tex           LaTeX table: latency + throughput per strategy
    │   └── interactions.tex      LaTeX table: improvement vs. baseline
    └── analysis/
        ├── dominance.csv         P95 pivot: strategy × (prompt_bucket, output_bucket)
        ├── interactions.csv      Per-strategy improvement relative to baseline
        └── separability.json     R² for prompt-only / output-only / combined feature sets
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'benchmark'`**  
Run from the project root, and ensure the three `__init__.py` files exist:
```bash
touch benchmark/__init__.py benchmark/analysis/__init__.py config/__init__.py
```

**`ModuleNotFoundError: No module named 'vllm'` in vLLM subprocess log**  
The vLLM subprocess inherits `sys.executable` from the calling Python. Make sure you activated the correct virtualenv before running:
```bash
source .venv/bin/activate
python main.py ...
```

**`RuntimeError: The NVIDIA driver on your system is too old`**  
PyTorch was built for a newer CUDA than your driver supports. Install the matching PyTorch wheel:
```bash
# Check your driver's max CUDA version
nvidia-smi | head -3
# Then install the right wheel (e.g. cu121 for CUDA 12.1/12.2)
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

**`ValueError: Bfloat16 is only supported on GPUs with compute capability >= 8.0`**  
V100 and older GPUs (compute capability 7.x) do not support bfloat16. The strategy definitions default to `float16` which is correct — if you see this error, check that your `config/strategies.py` has `dtype="float16"` in the `Strategy` dataclass default.

**`AttributeError: TokenizersBackend has no attribute all_special_tokens_extended`**  
`transformers` version is too new for vLLM 0.6.0. Pin compatible versions:
```bash
pip install "transformers==4.45.2" "tokenizers==0.20.3"
```

**`api_server.py: error: unrecognized arguments: --no-enable-log-requests`**  
Wrong flag for your vLLM version. vLLM 0.6.x uses `--disable-log-requests`. This is set automatically in `config/strategies.py` — make sure you are on vLLM 0.6.0.

**vLLM starts but SM utilization stays at 0% in nvtop**  
This is normal during startup — the model is loaded into VRAM but no requests have arrived yet. Once the benchmark runner begins sending requests you will see SM utilization rise to 80–95%.

**`OSError: libcudart.so.13: cannot open shared object file`**  
`torchaudio` was built for a different CUDA version than torch. Reinstall to match:
```bash
pip install torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

---

## Adding a New Strategy

In `config/strategies.py`, add an entry to `STRATEGIES`:

```python
"my_new_strategy": Strategy(
    name="my_new_strategy",
    tensor_parallel_size=1,
    enable_chunked_prefill=True,
    chunked_prefill_size=512,
    enable_speculation=True,
    ngram_prompt_lookup_max=3,
    num_speculative_tokens=5,
    dtype="float16",
),
```

Then add it to the relevant ablation group in `get_strategies_for_dimension()` if applicable.

---

## Adding a New Workload

In `config/workloads.py`, add an entry to `WORKLOADS`:

```python
"my_workload": WorkloadConfig(
    name="my_workload",
    task_type=TaskType.CODE,
    prompt_lengths=[512, 1024, 2048],
    output_lengths=[256, 512],
    arrival_pattern=ArrivalPattern.POISSON_HIGH,
    num_requests=200,
    target_rps=3.0,
),
```

---

## Validated Environment

| Component | Version | Notes |
|---|---|---|
| Python | 3.12 | 3.10 / 3.11 also supported |
| vLLM | 0.6.0 | Pin to this version |
| PyTorch | 2.4.0+cu121 | Match to your CUDA driver |
| transformers | 4.45.2 | Must pin — newer versions incompatible with vLLM 0.6.0 |
| tokenizers | 0.20.3 | Must pin alongside transformers |
| CUDA driver | 535.x (CUDA 12.2) | V100-SXM2-32GB validated |
| GPU | Tesla V100-32GB | A100 / H100 / L40S also supported |

---

## Citation

```bibtex
@inproceedings{llm_exec_benchmark_2027,
  title     = {Compiler-Inspired Execution Planning for LLM Request Dispatch},
  author    = {<authors>},
  booktitle = {Proceedings of the 49th International Conference on Software Engineering (ICSE)},
  year      = {2027}
}
```

---

## License

MIT
