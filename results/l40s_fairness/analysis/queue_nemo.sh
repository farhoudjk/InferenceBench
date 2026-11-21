#!/usr/bin/env bash
P=/home/ubuntu/projects/InferenceBench/results/l40s_fairness/_provenance
until grep -q QUEUE_DONE $P/progress.log 2>/dev/null; do sleep 60; done
until grep -q "snapshots/04d8a905" /tmp/claude-1000/-home-ubuntu-projects-InferenceBench/805375f3-5526-43b6-a5b1-ac3b176f446c/scratchpad/dl_nemo.log; do sleep 60; done
R=/home/ubuntu/projects/InferenceBench/results/l40s_fairness/analysis/run_one.sh
M=/home/ubuntu/.cache/huggingface/hub/models--mistralai--Mistral-Nemo-Instruct-2407/snapshots/04d8a90549d23fc6bd7f642064003592df51e9b3; K=mistral_nemo_12b
$R l40s_fair__${K}__vllm__default__c256_s256 vllm $M
$R l40s_fair__${K}__sglang__default__c256_s256 sglang $M
$R l40s_fair__${K}__tensorrt-llm__default__c256_s256 tensorrtllm $M
$R l40s_fair__${K}__tensorrt-llm__maxutil__c256_s256 tensorrtllm $M '{"scheduler_config":{"capacity_scheduler_policy":"MAX_UTILIZATION"}}'
$R l40s_fair__${K}__vllm__eqkv__c256_s256 vllm $M
$R l40s_fair__${K}__sglang__eqkv__c256_s256 sglang $M
$R l40s_fair__${K}__tensorrt-llm__eqkv__c256_s256 tensorrtllm $M
echo NEMO_QUEUE_DONE >> $P/progress.log
