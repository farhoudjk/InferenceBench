# L40S fairness experiments: TRT-LLM MAX_UTILIZATION (E1) and equal KV pools (E2)

Single L40S, fp16, TP=1, max context 8192, mixed_real (500 requests per rate), Poisson 2 and 8 req/s,
client concurrency C=256, scheduler limit S=256, timeouts 300 s TTFT / 600 s total, memory fraction 0.90.
Engines: vLLM 0.11.0, SGLang 0.5.5, TensorRT-LLM 1.0.0 (PyTorch backend). Run 2026-11-20/21.

## Summary

- **The comparison baseline is a November rerun of the defaults, not the July RQ1 runs.** The July environment
  no longer existed (see Deviations 1–3). In particular, the harness now sends prompts through each model's chat
  template, while July sent raw text. So the November workload produces about 40% fewer output tokens for the
  same requests, and its absolute numbers are not comparable to July's. Every variant below is compared with a
  default run from the same environment and the same code.
- **E1: TensorRT-LLM's TTFT tail comes from its default scheduling policy.** With MAX_UTILIZATION, TTFT_user p95
  drops from 190 s / 299 s (Llama, 2 / 8 rps) and 3.4 s / 107 s (Nemo) to 0.19–0.43 s, and strict goodput goes
  from 16–90% to 100% in all four cells. Output throughput rises 15–40%. TPOT rises slightly; the largest increase
  is Nemo at 8 rps, where p50 goes from 42 to 55 ms.
- **E2: equal KV pools do not change the picture.** vLLM and SGLang stay at 100% goodput with tails under 0.4 s
  with the smaller pool. TensorRT-LLM keeps its tail whatever its pool size: with the Nemo pool cut from 113,536 to
  103,504 tokens, goodput drops from 90% to 73% at 2 rps and stays at about 15% at 8 rps. For Llama the pools were
  already nearly equal, and TRT-LLM's own pool was POOL, so its eqkv run repeats its default.
- **The November defaults show the same ranking as July.** TensorRT-LLM under GUARANTEED_NO_EVICT is the only
  engine with large TTFT tails. In this workload vLLM and SGLang reach 100% goodput everywhere.

## Step 0: KV pool sizes (tokens)

| model | engine | July baseline log | Nov default log (low / high launch) | Nov eqkv log |
|---|---|---|---|---|
| llama31_8b | vLLM | 194,144 | 194,064 / 194,080 | 193,536 |
| llama31_8b | SGLang | 200,650 | 200,564 / 200,564 | 193,536 |
| llama31_8b | TensorRT-LLM | 193,472 | 193,536 / 193,536 | 193,536 |
| mistral_nemo_12b | vLLM | 103,568 | 103,504 / 103,504 | 103,504 |
| mistral_nemo_12b | SGLang | 109,038 | 108,969 / 108,969 | 103,504 |
| mistral_nemo_12b | TensorRT-LLM | 113,408 | 113,536 / 113,472 | 103,520 (3,235 × 32-token blocks) |

**POOL** (smallest Nov default pool, read from the low-rate launch logs): **llama31_8b = 193,536** (TensorRT-LLM),
**mistral_nemo_12b = 103,504** (vLLM). From July logs it would have been 193,472 and 103,568, a difference of at
most 4 blocks. Full table: `kv_pools.csv`.

Because POOL equals the engine's own default pool in two cases, those eqkv runs reproduce the default exactly:
**Llama TensorRT-LLM eqkv** and **Nemo vLLM eqkv**. Each is effectively a repeat of its default run, which gives
an estimate of run-to-run noise (see Notes on noise).

### Confirmation lines (low-rate launch; the high-rate launch shows the same values)

- vLLM, Llama, eqkv: `Overriding num_gpu_blocks=12130 with num_gpu_blocks_override=12096` and
  `GPU KV cache size: 193,536 tokens` (confirms 16-token blocks: 12,096 × 16)
- vLLM, Nemo, eqkv: `Overriding num_gpu_blocks=6469 with num_gpu_blocks_override=6469` and `GPU KV cache size: 103,504 tokens`
- SGLang, Llama, eqkv: `max_total_num_tokens=193536`. SGLang, Nemo, eqkv: `max_total_num_tokens=103504`
- TensorRT-LLM, Llama, eqkv: `Allocated 23.62 GiB for max tokens in paged KV cache (193536)`
- TensorRT-LLM, Nemo, eqkv: `Allocated 15.80 GiB for max tokens in paged KV cache (103520)` (max_tokens 103504, rounded up to whole 32-token blocks)
- TensorRT-LLM, maxutil: the LLM-API YAML passed via `--extra_llm_api_options` contains
  `scheduler_config: {capacity_scheduler_policy: MAX_UTILIZATION}` (`logs/trtllm_config_*__{low,high}.yaml`).
  **The TRT-LLM 1.0.0 server log does not print the capacity policy at the default log level, so there is no
  line in the launch log to quote** (Deviation 6). The evidence is the YAML the server was started with, plus the
  large change in behavior in E1 (the only setting that differs from the default run).

## E1: TensorRT-LLM, GUARANTEED_NO_EVICT (default) → MAX_UTILIZATION

Each cell is default → variant. All runs are from November.

| model | rps | TTFT p50 s (def → maxutil) | TTFT p95 s (def → maxutil) | TPOT p50 ms (def → maxutil) | TPOT p95 ms (def → maxutil) | tok/s (def → maxutil) | fin % (def → maxutil) | goodput % (def → maxutil) | peak KV % (def → maxutil) | max run (def → maxutil) | preempt (def → maxutil) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama31_8b | 2 | 0.11 → 0.10 | 189.73 → 0.19 | 35.4 → 33.9 | 44.1 → 47.7 | 595 → 686 | 100.0 → 100.0 | 60.0 → 100.0 | n/a → n/a | n/a → n/a | n/a → n/a |
| llama31_8b | 8 | 40.56 → 0.11 | 299.45 → 0.26 | 35.0 → 38.6 | 44.3 → 52.2 | 643 → 745 | 100.0 → 100.0 | 26.0 → 100.0 | n/a → n/a | n/a → n/a | n/a → n/a |
| mistral_nemo_12b | 2 | 0.11 → 0.11 | 3.37 → 0.25 | 39.3 → 40.7 | 43.3 → 45.8 | 285 → 356 | 100.0 → 100.0 | 90.0 → 100.0 | n/a → n/a | n/a → n/a | n/a → n/a |
| mistral_nemo_12b | 8 | 38.44 → 0.16 | 106.54 → 0.43 | 42.0 → 55.4 | 48.6 → 70.3 | 314 → 440 | 100.0 → 100.0 | 16.0 → 100.0 | n/a → n/a | n/a → n/a | n/a → n/a |

## E2: equal KV pool (POOL), default → eqkv

### vLLM
| model | rps | TTFT p50 s (def → eqkv) | TTFT p95 s (def → eqkv) | TPOT p50 ms (def → eqkv) | TPOT p95 ms (def → eqkv) | tok/s (def → eqkv) | fin % (def → eqkv) | goodput % (def → eqkv) | peak KV % (def → eqkv) | max run (def → eqkv) | preempt (def → eqkv) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama31_8b | 2 | 0.07 → 0.07 | 0.17 → 0.17 | 35.9 → 35.7 | 51.2 → 48.9 | 689 → 686 | 100.0 → 100.0 | 100.0 → 100.0 | 62.0 → 65.2 | 69 → 67 | 0 → 0 |
| llama31_8b | 8 | 0.08 → 0.08 | 0.22 → 0.27 | 40.8 → 40.1 | 55.0 → 53.7 | 924 → 906 | 100.0 → 100.0 | 100.0 → 100.0 | 90.3 → 75.1 | 146 → 142 | 0 → 0 |
| mistral_nemo_12b | 2 | 0.08 → 0.08 | 0.21 → 0.21 | 41.4 → 41.7 | 46.5 → 46.7 | 263 → 269 | 100.0 → 100.0 | 100.0 → 100.0 | 15.8 → 15.7 | 36 → 35 | 0 → 0 |
| mistral_nemo_12b | 8 | 0.11 → 0.12 | 0.36 → 0.39 | 60.4 → 60.7 | 78.8 → 79.8 | 399 → 445 | 100.0 → 100.0 | 100.0 → 100.0 | 56.3 → 58.3 | 142 → 145 | 0 → 0 |

### SGLang
| model | rps | TTFT p50 s (def → eqkv) | TTFT p95 s (def → eqkv) | TPOT p50 ms (def → eqkv) | TPOT p95 ms (def → eqkv) | tok/s (def → eqkv) | fin % (def → eqkv) | goodput % (def → eqkv) | peak KV % (def → eqkv) | max run (def → eqkv) | preempt (def → eqkv) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama31_8b | 2 | 0.09 → 0.08 | 0.19 → 0.18 | 36.9 → 38.3 | 51.6 → 53.5 | 710 → 717 | 100.0 → 100.0 | 100.0 → 100.0 | 61.9 → 70.5 | 70 → 74 | n/a → n/a |
| llama31_8b | 8 | 0.09 → 0.09 | 0.21 → 0.21 | 45.5 → 45.3 | 62.2 → 61.6 | 915 → 918 | 100.0 → 100.0 | 100.0 → 100.0 | 81.6 → 71.7 | 153 → 157 | n/a → n/a |
| mistral_nemo_12b | 2 | 0.11 → 0.11 | 0.25 → 0.25 | 42.2 → 42.2 | 48.9 → 49.2 | 276 → 324 | 100.0 → 100.0 | 100.0 → 100.0 | 14.6 → 16.4 | 37 → 38 | n/a → n/a |
| mistral_nemo_12b | 8 | 0.14 → 0.14 | 0.38 → 0.37 | 72.6 → 72.3 | 101.6 → 100.7 | 413 → 466 | 100.0 → 100.0 | 100.0 → 99.8 | 53.2 → 56.7 | 159 → 159 | n/a → n/a |

### TensorRT-LLM
| model | rps | TTFT p50 s (def → eqkv) | TTFT p95 s (def → eqkv) | TPOT p50 ms (def → eqkv) | TPOT p95 ms (def → eqkv) | tok/s (def → eqkv) | fin % (def → eqkv) | goodput % (def → eqkv) | peak KV % (def → eqkv) | max run (def → eqkv) | preempt (def → eqkv) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama31_8b | 2 | 0.11 → 0.18 | 189.73 → 180.36 | 35.4 → 37.2 | 44.1 → 43.9 | 595 → 609 | 100.0 → 100.0 | 60.0 → 52.6 | n/a → n/a | n/a → n/a | n/a → n/a |
| llama31_8b | 8 | 40.56 → 34.82 | 299.45 → 302.06 | 35.0 → 35.0 | 44.3 → 43.3 | 643 → 632 | 100.0 → 97.6 | 26.0 → 26.0 | n/a → n/a | n/a → n/a | n/a → n/a |
| mistral_nemo_12b | 2 | 0.11 → 0.12 | 3.37 → 19.16 | 39.3 → 40.3 | 43.3 → 44.0 | 285 → 292 | 100.0 → 100.0 | 90.0 → 73.2 | n/a → n/a | n/a → n/a | n/a → n/a |
| mistral_nemo_12b | 8 | 38.44 → 44.76 | 106.54 → 117.97 | 42.0 → 41.4 | 48.6 → 48.7 | 314 → 471 | 100.0 → 100.0 | 16.0 → 14.8 | n/a → n/a | n/a → n/a | n/a → n/a |

## Reference: July RQ1 defaults → November rerun defaults (same metric code)

These are **not like-for-like** (Deviations 1–3: chat template, environment, driver). They are shown only so every
new row has a baseline row computed by the same code. July's vLLM and SGLang tails at 2 rps (147–189 s) and the
July Nemo zero-token errors (34–36 per run) disappear in November. The likeliest reason is the chat template:
raw-text prompts made more requests run all the way to max_tokens.

| model | engine | rps | TTFT p50 s (Jul → Nov) | TTFT p95 s (Jul → Nov) | TPOT p50 ms (Jul → Nov) | TPOT p95 ms (Jul → Nov) | tok/s (Jul → Nov) | fin % (Jul → Nov) | goodput % (Jul → Nov) |
|---|---|---|---|---|---|---|---|---|---|
| llama31_8b | vllm | 2 | 0.09 → 0.07 | 188.68 → 0.17 | 46.9 → 35.9 | 277.8 → 51.2 | 714 → 689 | 92.6 → 100.0 | 84.4 → 100.0 |
| llama31_8b | vllm | 8 | 0.08 → 0.08 | 0.24 → 0.22 | 43.9 → 40.8 | 67.1 → 55.0 | 749 → 924 | 92.6 → 100.0 | 91.4 → 100.0 |
| llama31_8b | sglang | 2 | 0.10 → 0.09 | 147.19 → 0.19 | 49.6 → 36.9 | 72.0 → 51.6 | 615 → 710 | 94.0 → 100.0 | 78.2 → 100.0 |
| llama31_8b | sglang | 8 | 0.09 → 0.09 | 0.21 → 0.21 | 49.7 → 45.5 | 72.6 → 62.2 | 738 → 915 | 93.4 → 100.0 | 92.4 → 100.0 |
| llama31_8b | tensorrt-llm | 2 | 187.00 → 0.11 | 368.01 → 189.73 | 41.1 → 35.4 | 48.9 → 44.1 | 584 → 595 | 85.2 → 100.0 | 30.2 → 60.0 |
| llama31_8b | tensorrt-llm | 8 | 297.75 → 40.56 | 541.57 → 299.45 | 40.9 → 35.0 | 47.4 → 44.3 | 531 → 643 | 80.8 → 100.0 | 27.0 → 26.0 |
| mistral_nemo_12b | vllm | 2 | 0.08 → 0.08 | 0.22 → 0.21 | 42.3 → 41.4 | 50.7 → 46.5 | 271 → 263 | 93.2 → 100.0 | 93.2 → 100.0 |
| mistral_nemo_12b | vllm | 8 | 0.11 → 0.11 | 0.37 → 0.36 | 56.8 → 60.4 | 75.7 → 78.8 | 366 → 399 | 93.2 → 100.0 | 93.2 → 100.0 |
| mistral_nemo_12b | sglang | 2 | 0.10 → 0.11 | 0.23 → 0.25 | 42.0 → 42.2 | 50.4 → 48.9 | 308 → 276 | 93.2 → 100.0 | 93.2 → 100.0 |
| mistral_nemo_12b | sglang | 8 | 0.12 → 0.14 | 0.38 → 0.38 | 69.1 → 72.6 | 97.2 → 101.6 | 392 → 413 | 93.2 → 100.0 | 93.2 → 100.0 |
| mistral_nemo_12b | tensorrt-llm | 2 | 0.16 → 0.11 | 37.64 → 3.37 | 42.2 → 39.3 | 48.9 → 43.3 | 281 → 285 | 93.0 → 100.0 | 61.6 → 90.0 |
| mistral_nemo_12b | tensorrt-llm | 8 | 30.46 → 38.44 | 126.99 → 106.54 | 43.1 → 42.0 | 49.3 → 48.6 | 330 → 314 | 92.8 → 100.0 | 19.2 → 16.0 |

## Metric definitions (`analysis/compute_metrics.py`)

These follow the plan exactly, plus the following choices where the plan left a case open:
- finished = status "ok" and output_tokens > 0. Zero-token replies (status "error") are excluded from latency
  statistics and count as goodput misses.
- Requests that hit the 600 s total timeout after their first token (status "timeout", not ttft_timeout) keep
  their measured TTFT in the TTFT statistics. They are not finished.
- ttft_timeout requests count as TTFT 300 s + client_wait.
- Strict goodput: a finished request with output_tokens = 1 has no TPOT and passes the TPOT condition.
- Peak KV, max running and preemptions come from the per-request server-metric snapshots in requests.csv; the
  harness writes no server_metrics.csv. vLLM preemptions = max − min of the cumulative counter. SGLang exposes no
  preemption counter. TensorRT-LLM only gets one end-of-run snapshot (its /metrics endpoint long-polls), so all
  three are n/a for it.

## Notes on noise

The runs with identical configurations give a rough noise level:
- **Llama TensorRT-LLM eqkv vs default:** goodput 60.0 vs 52.6% at 2 rps; at 8 rps, 12 ttft timeouts vs 0;
  TTFT p95 within 5%.
- **Nemo vLLM eqkv vs default:** all latency metrics within about 3%, but throughput 399 vs 445 tok/s at 8 rps.

The throughput definition divides by the span up to the last finishing request, so a single long tail request
moves it by 10% or more. Treat throughput differences under about 15% as noise. Likewise TRT-LLM goodput
differences under about 10 points.

## Runs and checks (Step 3)

All 14 run IDs `l40s_fair__<model>__<engine>__<variant>__c256_s256` completed both rates: 500/500 requests per
rate, no crash, engine log, harness log and command line saved for each of the 28 launches. Every request
finished, except **Llama TensorRT-LLM eqkv at 8 rps, which had 12 ttft_timeouts**. The harness writes no audit.

## Deviations from the plan, and failed runs

1. **Environment rebuilt.** The July vLLM venv (`InferenceBench/.venv`) and TRT-LLM venv (`.venv-trtllm`) had
   been deleted, and both models and the mixed_real source files were gone from the HF cache. vLLM 0.11.0 and
   SGLang 0.5.5 come from `~/projects/Bench2/.venv` (the venv the July SGLang baseline used). TRT-LLM 1.0.0 was
   reinstalled fresh with torch 2.7.1+cu128, with **pyarrow pinned to <21**: pip pulled pyarrow 26, which needs
   NumPy 2, while TRT pins NumPy 1.26. Models: Llama-3.1-8B-Instruct at rev 0e9e39f (July's revision was not
   recorded); Nemo at the same snapshot as July (04d8a905). Driver 580.178.04 / CUDA 13.0, against 570 / 12.8 in
   July. Versions, freezes and nvidia-smi output are in `_provenance/`.
2. **Defaults rerun (12 extra rate-runs, user decision).** Because of 1 and 3, the comparison baseline is the
   November rerun; July rows are reported for reference only.
3. **Harness code differs from July.** `benchmark/runner.py` and the engine adapters were edited after the July
   runs (Jul 30–Aug 3) and are uncommitted. The relevant change is that `runner.py` now applies the model's chat
   template to every prompt. For the same request IDs, the November runs generate about 43% fewer output tokens
   (Llama vLLM, 2 rps, requests that succeeded in both: 324k vs 573k). The trace itself (request IDs, prompt
   lengths, target output lengths, arrival times) matches July exactly at both rates (checked request by request).
   That match requires `--num-requests 500`, because the workload default is now 400.
4. **One server launch per rate, not per run.** `main.py` starts and stops the engine for each rate config,
   exactly as in the July baselines. Each launch's engine log is kept as `logs/<engine>_..._seqs256__{low,high}.log`.
5. **POOL taken from the November default logs, not the July logs** (it differs by at most 4 blocks), so that
   eqkv is compared with defaults from the same environment.
6. **TRT-LLM log format.** There is no "primary blocks=… tokens per block=…" line in 1.0.0, so I used
   `Allocated … GiB for max tokens in paged KV cache (N)` (32 tokens per block). The capacity policy is not
   printed (see Confirmation lines).
7. **TRT-LLM config file** is `<run>/trtllm_config_<strategy>_<port>.yaml` (kept per rate as
   `logs/trtllm_config_*__{low,high}.yaml`), not `trtllm_cfg/extra_*.json`.
8. **No server_metrics.csv, run_config.json or audit** in this harness; see Metric definitions. The configuration
   used is in `_provenance/settings.txt` and `_provenance/launches/*.cmd`.
9. **Two opt-in harness changes** (uncommitted, like the rest of the harness; snapshot in
   `_provenance/harness_snapshot/`, full working-tree diff in `_provenance/git_diff_working_tree_vs_HEAD.patch`):
   - `benchmark/engines/tensorrtllm_engine.py`: the env var `TRTLLM_EXTRA_LLM_API_OPTIONS` (JSON) is
     deep-merged into the LLM-API YAML. When it is unset, every other run is unchanged.
   - `config/workloads.py`: an opt-in on-disk cache (`INFERENCEBENCH_DATASET_CACHE`) of the dataset loaders'
     output. It stops each launch from re-tokenizing the datasets for about 4 minutes while the GPU sits idle.
     The trace with the cache was re-verified identical to July's. Enabled from the Llama SGLang default
     high-rate launch onward.
10. **Failed attempt (attempt 1, 2026-11-20 ~20:05Z).** Llama sglang/default, tensorrt-llm/default and
    tensorrt-llm/maxutil failed at server start: SGLang because `ninja` was not on PATH (fixed by putting the
    venv's bin/ on PATH), TRT-LLM because of the pyarrow/NumPy conflict (Deviation 1). The three Llama eqkv runs
    were skipped (no POOL available), and Nemo sglang/default was stopped during startup. No requests were served
    in any of them. Their logs are in `_provenance/failed_attempts/attempt1/`, and all were rerun once with the
    fixes. Nemo vLLM default ran during this attempt and succeeded; vLLM was unaffected by either bug, so it was
    kept.
11. **Disk incidents.** Twice the disk filled up (the second time, TRT-LLM had downloaded Llama's unused
    `original/consolidated.00.pth`). The Nemo download failed, and the GPU sat idle 23:01–00:10Z before it was
    restarted. No run was affected. Details are in `_provenance/failures.log`.
12. With user approval, the HF-cache copies of Qwen3-30B-A3B-FP8 and OLMoE-1B-7B were deleted to make room. The
    Llama weights were deleted after the Llama runs.

## Files

- `l40s_fair__*/`: requests.csv, gpu_samples (if present), summary.json, logs/ (engine and harness logs per
  rate, TRT YAMLs, GPU-process checks), report figures/tables generated by the harness
- `metrics.csv`: Step 4 metrics; `variant` is one of {default, maxutil, eqkv}, `source` is one of
  {baseline_jul2026, rerun_nov2026}
- `kv_pools.csv`: Step 0
- `analysis/`: compute_metrics.py, kv_pools.py, make_tables.py, run_one.sh, queue_*.sh
- `_provenance/`: engine versions, pip freezes, nvidia-smi -q, git HEAD/status/diff, harness snapshot, per-launch
  command lines, settings, progress/failure logs (dataset_cache/ is not committed; it is regenerable)
