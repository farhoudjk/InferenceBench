#!/usr/bin/env bash
P=/home/ubuntu/projects/InferenceBench/results/l40s_fairness/_provenance
until grep -q LLAMA_DONE $P/progress.log; do sleep 3; done
python3 - <<'PY'
import os,shutil
hub=os.path.expanduser("~/.cache/huggingface/hub"); m=os.path.join(hub,"models--meta-llama--Llama-3.1-8B-Instruct")
for root,_,fs in os.walk(os.path.join(m,"snapshots")):
    for f in fs:
        if f.endswith(".safetensors"):
            p=os.path.realpath(os.path.join(root,f))
            if p.startswith(hub+"/blobs/") and os.path.exists(p): os.remove(p)
print("llama weights removed (tokenizer kept for workload token counting)")
PY
