"""Fairness-experiment metrics (L40S, mixed_real, C=256, S=256, Poisson 2 & 8 rps).

Same code for the July 2026 baseline runs and the Nov 2026 runs.
Usage: python compute_metrics.py  -> writes ../metrics.csv
"""
import glob
import math
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)                       # results/l40s_fairness
RESULTS = os.path.dirname(OUT)                    # results/
TTFT_TIMEOUT_S = 300.0
GOOD_TTFT_S, GOOD_TPOT_S = 2.0, 0.200
RATES = {"low_poisson": 2.0, "high_poisson": 8.0}
ENGINE_NAME = {"vllm": "vllm", "sglang": "sglang", "tensorrtllm": "tensorrt-llm"}

BASELINES = {  # July 2026 RQ1 runs (read-only)
    ("llama31_8b", "vllm"): "c256_mixed_vllm_seqs256",
    ("llama31_8b", "sglang"): "c256_mixed_sglang_seqs256",
    ("llama31_8b", "tensorrt-llm"): "c256_mixed_trtllm_seqs256",
    ("mistral_nemo_12b", "vllm"): "nemo_sweep_mixed256_vllm_seqs256",
    ("mistral_nemo_12b", "sglang"): "nemo_sweep_mixed256_sglang_seqs256",
    ("mistral_nemo_12b", "tensorrt-llm"): "nemo_sweep_mixed256_trtllm_seqs256",
}


def pct(s, q):
    s = s.dropna()
    return float(np.percentile(s, q)) if len(s) else math.nan


def metrics(df: pd.DataFrame, engine: str) -> dict:
    n = len(df)
    err = df["error"].fillna("").astype(str)
    finished = (df["status"] == "ok") & (df["output_tokens"] > 0)
    # Zero-token replies = status "error" (e.g. "no tokens received"). Total-timeout rows
    # (status "timeout", not ttft_timeout) did receive a first token -> real ttft_s kept.
    zero_tok = df["status"] == "error"
    ttft_to = (df["status"] == "timeout") & err.str.startswith("ttft_timeout")

    send0, arr0 = df["send_time"].min(), df["arrival_time"].min()
    cw = ((df["send_time"] - send0) - (df["arrival_time"] - arr0)).clip(lower=0)

    ttft = df["ttft_s"].copy()
    ttft[ttft_to] = TTFT_TIMEOUT_S
    ttft_user = (ttft + cw)[~zero_tok]           # zero-token replies excluded from latency stats

    fin = df[finished]
    tpot = ((fin["e2e_s"] - fin["ttft_s"]) / (fin["output_tokens"] - 1)).where(fin["output_tokens"] > 1)
    t_end = (fin["send_time"] + fin["e2e_s"]).max()
    t_start = (df["send_time"] - cw).min()
    thr = fin["output_tokens"].sum() / (t_end - t_start) if len(fin) else 0.0

    # Strict goodput over ALL requests; single-token outputs have no decode phase -> pass TPOT.
    good = finished & ((df["ttft_s"] + cw) <= GOOD_TTFT_S) & (tpot.reindex(df.index).fillna(0) <= GOOD_TPOT_S)

    out = {
        "n_requests": n, "n_finished": int(finished.sum()), "n_ttft_timeout": int(ttft_to.sum()),
        "n_zero_token_or_error": int(zero_tok.sum()),
        "n_total_timeout": int(((df["status"] == "timeout") & ~ttft_to).sum()),
        "ttft_user_p50_s": pct(ttft_user, 50), "ttft_user_p95_s": pct(ttft_user, 95),
        "tpot_p50_ms": pct(tpot, 50) * 1000, "tpot_p95_ms": pct(tpot, 95) * 1000,
        "throughput_tok_s": thr, "finished_pct": 100 * finished.mean(),
        "strict_goodput_pct": 100 * good.mean(),
        "client_wait_max_s": float(cw.max()),
    }
    # Server metrics: per-request snapshots in requests.csv (no server_metrics.csv in this harness).
    # TensorRT-LLM only has one end-of-run snapshot -> not a peak, reported as NaN.
    if engine == "tensorrt-llm":
        out.update(peak_kv_pct=math.nan, max_running=math.nan, preemptions=math.nan)
    else:
        kv = df["kv_cache_usage_pct"].dropna()
        out["peak_kv_pct"] = float(kv.max() * (100 if kv.max() <= 1.0 else 1)) if len(kv) else math.nan
        run = df["num_running_seqs"].dropna()
        out["max_running"] = float(run.max()) if len(run) else math.nan
        pre = df["num_preemptions"].dropna()
        out["preemptions"] = float(pre.max() - pre.min()) if (engine == "vllm" and len(pre)) else math.nan
    return out


def rows_for(path, model, engine, variant, source, run_id):
    df = pd.read_csv(path, low_memory=False)
    rows = []
    for rl, rps in RATES.items():
        sub = df[df["rate_label"] == rl].reset_index(drop=True)
        if sub.empty:
            continue
        rows.append({"model": model, "engine": engine, "variant": variant, "source": source,
                     "rate_rps": rps, "run_id": run_id, **metrics(sub, engine)})
    return rows


def main():
    rows = []
    for (model, engine), rid in BASELINES.items():
        rows += rows_for(os.path.join(RESULTS, rid, "requests.csv"), model, engine, "default",
                         "baseline_jul2026", rid)
    for d in sorted(glob.glob(os.path.join(OUT, "l40s_fair__*"))):
        rid = os.path.basename(d)
        m = re.match(r"l40s_fair__(.+?)__(.+?)__(.+?)__c256_s256$", rid)
        f = os.path.join(d, "requests.csv")
        if m and os.path.exists(f):
            rows += rows_for(f, m.group(1), m.group(2), m.group(3), "rerun_nov2026", rid)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "metrics.csv"), index=False, float_format="%.4f")
    print(out.to_string(max_cols=30))


if __name__ == "__main__":
    main()
