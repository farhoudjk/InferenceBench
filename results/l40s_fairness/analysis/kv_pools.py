"""Extract each engine's KV pool size (tokens) from launch logs.

python kv_pools.py               -> writes ../kv_pools.csv (July baseline + Nov runs)
python kv_pools.py --pool MODEL  -> prints POOL = min over the Nov *default* launches of MODEL
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
RESULTS = os.path.dirname(OUT)
sys.path.insert(0, HERE)
from compute_metrics import BASELINES  # noqa: E402

PATTERNS = {
    "vllm": r"GPU KV cache size: ([\d,]+) tokens",
    "sglang": r"max_total_num_tokens=(\d+)",
    "tensorrt-llm": r"Allocated [\d.]+ GiB for max tokens in paged KV cache \((\d+)\)",
}


def pool_from_log(path, engine):
    vals = re.findall(PATTERNS[engine], open(path, errors="replace").read())
    # TRT-LLM logs a first tiny dry-run allocation (8256 tokens) before the real one: take the last.
    return int(vals[-1].replace(",", "")) if vals else None


def engine_logs(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, "logs", "*_baseline_full_fp16_tp1_seqs256__*.log"))) or \
        sorted(glob.glob(os.path.join(run_dir, "*_baseline_full_fp16_tp1_seqs256.log")))


def rows():
    out = []
    for (model, engine), rid in BASELINES.items():
        for lg in engine_logs(os.path.join(RESULTS, rid)):
            out.append((model, engine, "default", "baseline_jul2026", rid, os.path.basename(lg), pool_from_log(lg, engine)))
    for d in sorted(glob.glob(os.path.join(OUT, "l40s_fair__*"))):
        _, model, engine, variant, _ = os.path.basename(d).split("__")
        for lg in engine_logs(d):
            out.append((model, engine, variant, "rerun_nov2026", os.path.basename(d), os.path.basename(lg), pool_from_log(lg, engine)))
    return out


def nov_pool(model):
    vals = {}
    for m, e, v, s, rid, lg, p in rows():
        if m == model and v == "default" and s == "rerun_nov2026" and p:
            vals.setdefault(e, p)
    if set(vals) != set(PATTERNS):
        raise SystemExit(f"missing Nov default pools for {model}: {vals}")
    return min(vals.values()), vals


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--pool":
        print(nov_pool(sys.argv[2])[0])
    else:
        import csv
        with open(os.path.join(OUT, "kv_pools.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "engine", "variant", "source", "run_id", "log", "pool_tokens"])
            for r in rows():
                w.writerow(r)
                print(*r, sep=",")
