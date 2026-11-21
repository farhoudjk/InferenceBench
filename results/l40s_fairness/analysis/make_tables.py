"""Markdown tables for REPORT.md from ../metrics.csv."""
import os, pandas as pd
d = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "metrics.csv"))
M = [("ttft_user_p50_s", "TTFT p50 s", 2), ("ttft_user_p95_s", "TTFT p95 s", 2), ("tpot_p50_ms", "TPOT p50 ms", 1),
     ("tpot_p95_ms", "TPOT p95 ms", 1), ("throughput_tok_s", "tok/s", 0), ("finished_pct", "fin %", 1),
     ("strict_goodput_pct", "goodput %", 1), ("peak_kv_pct", "peak KV %", 1), ("max_running", "max run", 0), ("preemptions", "preempt", 0)]
def f(v, n): return "n/a" if pd.isna(v) else f"{v:.{n}f}"
def table(engine, variant, source="rerun_nov2026", base_source="rerun_nov2026"):
    hdr = "| model | rps | " + " | ".join(f"{h} (def → {variant})" for _, h, _ in M) + " |"
    out = [hdr, "|" + "---|" * (len(M) + 2)]
    for model in ["llama31_8b", "mistral_nemo_12b"]:
        for rps in [2.0, 8.0]:
            q = (d.model == model) & (d.engine == engine) & (d.rate_rps == rps)
            a = d[q & (d.variant == "default") & (d.source == base_source)]; b = d[q & (d.variant == variant) & (d.source == source)]
            if a.empty or b.empty: continue
            a, b = a.iloc[0], b.iloc[0]
            out.append(f"| {model} | {rps:g} | " + " | ".join(f"{f(a[c], n)} → {f(b[c], n)}" for c, _, n in M) + " |")
    return "\n".join(out)
def july():
    out = ["| model | engine | rps | " + " | ".join(f"{h} (Jul → Nov)" for _, h, _ in M[:7]) + " |", "|" + "---|" * 10]
    for model in ["llama31_8b", "mistral_nemo_12b"]:
        for eng in ["vllm", "sglang", "tensorrt-llm"]:
            for rps in [2.0, 8.0]:
                q = (d.model == model) & (d.engine == eng) & (d.rate_rps == rps) & (d.variant == "default")
                a, b = d[q & (d.source == "baseline_jul2026")].iloc[0], d[q & (d.source == "rerun_nov2026")].iloc[0]
                out.append(f"| {model} | {eng} | {rps:g} | " + " | ".join(f"{f(a[c], n)} → {f(b[c], n)}" for c, _, n in M[:7]) + " |")
    return "\n".join(out)
if __name__ == "__main__":
    print("### E1 TRT-LLM maxutil\n" + table("tensorrt-llm", "maxutil"))
    for e in ["vllm", "sglang", "tensorrt-llm"]:
        print(f"\n### E2 eqkv {e}\n" + table(e, "eqkv"))
    print("\n### July vs Nov defaults\n" + july())
