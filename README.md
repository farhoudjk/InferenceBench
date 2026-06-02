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

## What This Benchmark Measures

### Execution Strategy Space

Each strategy is a configuration tuple across five independently variable dimensions:

| Dimension | Options |
|---|---|
| Prefill mode | Full prefill, Chunked-256, Chunked-512, Chunked-1024 |
| Tensor parallelism | TP-1, TP-2, TP-4 |
| Speculative decoding | None, ngram-3, ngram-5 |
| Quantization | fp16, fp8 |
| Combinations | Chunked + speculation interaction |

12 strategies are defined by default (fractional factorial design — avoids full cross-product).

### Workload Space

| Workload | Task type | Prompt lengths | Output lengths | Arrival pattern |
|---|---|---|---|---|
| `chat_poisson_low` | Chat | 128–1024 | 64–256 | Poisson λ=1.0 rps |
| `chat_poisson_high` | Chat | 128–1024 | 64–256 | Poisson λ=4.0 rps |
| `code_uniform` | Code generation | 256–2048 | 256–1024 | Uniform 1.5 rps |
| `summarization_long_prompt` | Summarization | 2048–8192 | 128–512 | Poisson λ=0.5 rps |
| `reasoning_bursty` | CoT reasoning | 256–1024 | 512–1024 | Bursty (8 req/burst) |
| `mixed_all` | Mixed | 128–4096 | 64–1024 | Poisson λ=2.0 rps |

### Metrics Collected Per Request

- **TTFT** (Time to First Token) — via SSE streaming, millisecond resolution
- **E2E latency** — full request completion time
- **Tokens per second** — decode throughput
- **KV cache utilization** — from vLLM Prometheus `/metrics`
- **Preemption count** — number of KV evictions during the run
- **GPU utilization, memory, power** — 1 Hz background sampling via pynvml

### Research Questions Answered

| RQ | Question | Output |
|---|---|---|
| RQ1 | Which strategy has lowest TTFT P95 per (prompt_len, output_len) regime? | `figures/dominance_heatmap.pdf` |
| RQ2 | Are strategy costs separable across request dimensions? | `analysis/separability.json` |
| RQ3 | How accurately does a fitted cost model predict per-strategy latency? | Cost model MAPE per strategy |
| RQ4 | What are interaction effects between combined strategies? | `figures/ttft_cdf.pdf`, `tables/interactions.tex` |
| RQ5 | What is the overhead of plan selection at request admission time? | Addressed in Paper 2 |

---

## Project Structure

```
llm_exec_benchmark/
├── main.py                         Entry point and CLI orchestrator
├── requirements.txt
├── config/
│   ├── strategies.py               Strategy definitions + vLLM CLI arg builder
│   └── workloads.py                Workload configs + Poisson/bursty trace generator
└── benchmark/
    ├── launcher.py                 vLLM subprocess lifecycle (start, health-poll, teardown)
    ├── runner.py                   Async trace replayer with SSE streaming + TTFT measurement
    ├── gpu_monitor.py              Background GPU sampler (pynvml / nvidia-smi fallback)
    ├── storage.py                  CSV + JSON persistence per run
    └── analysis/
        ├── cost_model.py           Separability analysis, dominance pivot, GBT cost model fitting
        └── report.py               Paper-quality figures (PDF) + LaTeX tables
```

---

## Hardware Requirements

| Configuration | Supported strategies |
|---|---|
| 1 × GPU (A100 / H100 / L40S) | All TP-1 strategies (10 of 12) |
| 2 × GPU | All TP-1 + TP-2 strategies (11 of 12) |
| 4 × GPU | Full strategy space including TP-4 |

The launcher automatically skips strategies that require more GPUs than available.  
For a single-GPU setup, simply omit TP-4 — TP-1 vs TP-2 is the meaningful comparison anyway.

Recommended minimum: **40 GB VRAM** for Llama-3.1-8B at fp16 with max context 8192.

---

## Installation

```bash
# 1. Clone and enter project
git clone <repo>
cd llm_exec_benchmark

# 2. Install benchmark dependencies
pip install -r requirements.txt

# 3. Install vLLM (install separately — version must match your CUDA)
pip install vllm>=0.4.0

# 4. Verify GPU access
nvidia-smi
python -c "import vllm; print(vllm.__version__)"
```

---

## Running the Benchmark

### Quickstart — single strategy × workload (sanity check, ~10 min)

```bash
python main.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --strategy baseline_full_fp16_tp1 \
  --workload chat_poisson_low
```

### Ablation over one dimension (~1–2 hours per dimension)

Varies a single dimension while holding others constant. Use this to answer RQ1/RQ2/RQ4 per dimension.

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

### Multiple workloads in one run

```bash
python main.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dimension chunked_prefill \
  --workloads chat_poisson_low chat_poisson_high summarization_long_prompt
```

### Full benchmark — all strategies × all workloads (~8–12 hours on A100)

```bash
python main.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --run-all \
  --gpu-indices 0,1          # omit for single GPU
```

### Multi-GPU runs (TP-2 / TP-4)

```bash
# 2 GPUs
python main.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --run-all \
  --gpu-indices 0,1

# 4 GPUs
python main.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --run-all \
  --gpu-indices 0,1,2,3
```

### Regenerate report from existing results (no re-running)

```bash
python main.py \
  --model placeholder \
  --report-only \
  --run-id 20260526_120000
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
| `--no-gpu-monitor` | off | Disable GPU sampling (faster) |
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
    └── tables/
        ├── summary.tex           LaTeX table: latency + throughput per strategy
        └── interactions.tex      LaTeX table: improvement vs. baseline
    └── analysis/
        ├── dominance.csv         P95 pivot: strategy × (prompt_bucket, output_bucket)
        ├── interactions.csv      Per-strategy improvement relative to baseline
        └── separability.json     R² for prompt-only / output-only / combined feature sets
```

---

## Adding a New Strategy

In `config/strategies.py`, add an entry to the `STRATEGIES` dict:

```python
"my_new_strategy": Strategy(
    name="my_new_strategy",
    tensor_parallel_size=1,
    enable_chunked_prefill=True,
    chunked_prefill_size=512,
    enable_speculation=True,
    ngram_prompt_lookup_max=3,
    num_speculative_tokens=5,
),
```

Then add it to the relevant ablation group in `get_strategies_for_dimension()` if applicable.

---

## Adding a New Workload

In `config/workloads.py`, add an entry to the `WORKLOADS` dict:

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

| Component | Version |
|---|---|
| Python | 3.10 / 3.11 / 3.12 |
| vLLM | ≥ 0.4.0 |
| CUDA | 12.1+ |
| GPU | A100-40GB / A100-80GB / H100 / L40S |

---

## Citation

If you use this benchmark in your research, please cite:

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
