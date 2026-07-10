# Known Issues & Fixes (2026-07-10)

Summary of problems found while trying to run **parallel** benchmark jobs with
chunked-prefill strategies on this host (two **Tesla V100-SXM2-32GB**, Volta,
compute capability 7.0; vLLM 0.6.0, Triton 3.0.0).

---

## 1. Config bug — chunked prefill flag was never sent ✅ FIXED

**Symptom:** every chunked run crashed at server startup with
```
ValueError: max_num_batched_tokens (256) is smaller than max_model_len (8192).
```

**Cause:** the `chunked_*` strategies set a small `max_num_batched_tokens`
(256/512/1024) but left `enable_chunked_prefill=False`. On **vLLM 0.6.0** (this
host), chunked prefill is OFF by default and the scheduler rejects
`max_num_batched_tokens < max_model_len` unless it is explicitly enabled. The old
code comments assumed vLLM 0.11 (where the flag is a no-op), which is how this slipped in.

**Fix:** added `enable_chunked_prefill=True` to all six `chunked_*` strategies in
`config/strategies.py`; corrected the outdated 0.11 comment.

---

## 2. Parallel jobs killed each other ✅ FIXED

**Symptom:** running two jobs at once, each died with the other's server being
killed: `Killed stale vLLM server PID <n>`. Empty/338-byte logs.

**Cause (two separate defects):**
- `kill_stale_gpu_workers()` (`benchmark/launcher.py`) did a **machine-global**
  sweep — `pgrep -f vllm.entrypoints.openai.api_server` + `nvidia-smi` — and
  SIGKILLed *every* vLLM server / GPU worker except its own `main.py` PID. The
  second job to launch murdered the first job's server.
- **`CUDA_VISIBLE_DEVICES` was never set.** `--gpu-indices` only fed diagnostic
  logging and the TP-count guard, so vLLM always defaulted to GPU 0 — both jobs
  piled onto the same GPU.

**Fix:**
- `VLLMLauncher` now takes `gpu_indices` and pins each subprocess via
  `env["CUDA_VISIBLE_DEVICES"]`.
- All cleanup is now scoped: server frontends matched by their `--port` (via
  `/proc/<pid>/cmdline`), GPU workers matched by physical GPU UUID (nvidia-smi
  index→uuid map). Every caller (main.py startup + atexit/signal, tui.py, launcher
  start/stop) passes the job's own GPU/port scope.

**How to run in parallel:** each job MUST get a **distinct `--gpu-indices` AND a
distinct `--port`** (both default to GPU 0 / port 8100). Example: llama on
`--gpu-indices 0 --port 8100`, deepseek on `--gpu-indices 1 --port 8101`.

---

## 3. Chunked prefill cannot run on V100 (Volta) ⚠️ HARDWARE LIMIT — GUARDED

**Symptom:** after fixes #1, the server boots (`Uvicorn running`, `/health 200
OK`) but the first chunked forward pass aborts the GPU worker with:
```
mma -> mma layout conversion is only supported on Ampere
```

**Cause:** chunked prefill's prefix-attention path
(`PagedAttention.forward_prefix` → `context_attention_fwd`, a **Triton** kernel in
`vllm/attention/ops/prefix_prefill.py`) emits MMA→MMA layout conversions that
Triton only compiles for **Ampere+ (compute capability ≥ 8.0)**. V100 is Volta
(7.0). The XFormers fallback only covers pure-prefill / pure-decode; any chunked
step with a cached prefix goes through the Triton kernel regardless, and vLLM 0.6.0
ships **no** non-Triton alternative for this path. There is no configuration escape.

Note: this is **not** a general "Volta can't do attention" problem — baseline
attention runs fine on V100 via XFormers. It is specifically that the only
chunked-prefill kernel vLLM provides has no Volta-compatible build.

**Impact — none of the chunked strategies are runnable on this host:**

| Strategy | Dimension(s) | V100 |
|---|---|---|
| `chunked_256_fp16_tp1` | chunked_prefill | ❌ |
| `chunked_512_fp16_tp1` | chunked_prefill, interaction | ❌ |
| `chunked_1024_fp16_tp1` | chunked_prefill | ❌ |
| `chunked_512_fp16_tp2` | tensor_parallel | ❌ |
| `chunked_512_fp8_tp1` | quantization | ❌ |
| `chunked_512_spec_ngram3_fp16_tp1` | interaction | ❌ |

Per dimension: `chunked_prefill` → only baseline runs; `tensor_parallel` → 3/4 run;
`quantization` → 2/3 run; `interaction` → 2/4 run; `speculation` → all run.

**Mitigation:** added `_check_chunked_prefill_support` in `benchmark/launcher.py`
(mirrors the existing AWQ guard). Chunked strategies on a non-Ampere GPU now **fail
fast with a clear message** instead of loading the model and dying on a cryptic
Triton assertion. This is per-strategy, so a multi-strategy sweep still runs its
non-chunked strategies and produces data.

**To actually benchmark chunked prefill:** run those strategies on an **Ampere or
newer** GPU (A100, A6000, RTX 30/40, etc.). Workarounds on Volta (rewriting the
Triton kernel without tensor cores, or downgrading Triton) are impractical and
would produce non-representative numbers, so they are not pursued.

---

## Runnable on V100 today

Non-chunked strategies work: `baseline_full_fp16_tp1/tp2/tp4`,
`baseline_full_fp8_tp1`, `spec_ngram3_fp16_tp1`, `spec_ngram5_fp16_tp1`.
