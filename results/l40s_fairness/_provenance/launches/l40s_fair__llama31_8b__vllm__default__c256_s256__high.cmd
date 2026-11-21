2026-11-20T19:52:01Z
TRTLLM_EXTRA_LLM_API_OPTIONS=
/home/ubuntu/projects/Bench2/.venv/bin/python main.py --model meta-llama/Llama-3.1-8B-Instruct --engine vllm --results-dir results/l40s_fairness --run-id l40s_fair__llama31_8b__vllm__default__c256_s256 --strategy baseline_full_fp16_tp1 --workload mixed_real --num-requests 500 --rates high --arrival-patterns poisson --max-concurrent 256 --max-num-seqs-override 256 --ttft-timeout 300 --request-timeout 600 --seed 42 
