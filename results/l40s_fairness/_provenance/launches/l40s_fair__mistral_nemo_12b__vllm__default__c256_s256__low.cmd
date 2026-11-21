2026-11-20T20:07:57Z
TRTLLM_EXTRA_LLM_API_OPTIONS=
/home/ubuntu/projects/Bench2/.venv/bin/python main.py --model /home/ubuntu/.cache/huggingface/hub/models--mistralai--Mistral-Nemo-Instruct-2407/snapshots/04d8a90549d23fc6bd7f642064003592df51e9b3 --engine vllm --results-dir results/l40s_fairness --run-id l40s_fair__mistral_nemo_12b__vllm__default__c256_s256 --strategy baseline_full_fp16_tp1 --workload mixed_real --num-requests 500 --rates low --arrival-patterns poisson --max-concurrent 256 --max-num-seqs-override 256 --ttft-timeout 300 --request-timeout 600 --seed 42 
