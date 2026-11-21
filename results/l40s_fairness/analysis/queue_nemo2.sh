#!/usr/bin/env bash
R=/home/ubuntu/projects/InferenceBench/results/l40s_fairness/analysis/run_one.sh
P=/home/ubuntu/projects/InferenceBench/results/l40s_fairness/_provenance
MU='{"scheduler_config":{"capacity_scheduler_policy":"MAX_UTILIZATION"}}'
for i in 1 2 3; do
/home/ubuntu/projects/Bench2/.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('mistralai/Mistral-Nemo-Instruct-2407', revision='04d8a90549d23fc6bd7f642064003592df51e9b3', allow_patterns=['*.json','model-*.safetensors','tokenizer*'])" > $P/dl_nemo.log 2>&1 && break
done || { echo "nemo download failed x3" >> $P/failures.log; exit 1; }
M=/home/ubuntu/.cache/huggingface/hub/models--mistralai--Mistral-Nemo-Instruct-2407/snapshots/04d8a90549d23fc6bd7f642064003592df51e9b3; K=mistral_nemo_12b
$R l40s_fair__${K}__vllm__default__c256_s256 vllm $M
$R l40s_fair__${K}__sglang__default__c256_s256 sglang $M
$R l40s_fair__${K}__tensorrt-llm__default__c256_s256 tensorrtllm $M
$R l40s_fair__${K}__tensorrt-llm__maxutil__c256_s256 tensorrtllm $M "$MU"
$R l40s_fair__${K}__vllm__eqkv__c256_s256 vllm $M
$R l40s_fair__${K}__sglang__eqkv__c256_s256 sglang $M
$R l40s_fair__${K}__tensorrt-llm__eqkv__c256_s256 tensorrtllm $M
echo ALL_DONE >> $P/progress.log
