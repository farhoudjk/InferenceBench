#!/usr/bin/env bash
R=/home/ubuntu/projects/InferenceBench/results/l40s_fairness/analysis/run_one.sh
M=meta-llama/Llama-3.1-8B-Instruct; K=llama31_8b; POOL=193472
$R l40s_fair__${K}__vllm__default__c256_s256 vllm $M
$R l40s_fair__${K}__sglang__default__c256_s256 sglang $M
$R l40s_fair__${K}__tensorrt-llm__default__c256_s256 tensorrtllm $M
$R l40s_fair__${K}__tensorrt-llm__maxutil__c256_s256 tensorrtllm $M '{"scheduler_config":{"capacity_scheduler_policy":"MAX_UTILIZATION"}}'
$R l40s_fair__${K}__vllm__eqkv__c256_s256 vllm $M - "--num-gpu-blocks-override=$((POOL/16))"
$R l40s_fair__${K}__sglang__eqkv__c256_s256 sglang $M - "--max-total-tokens=$POOL"
$R l40s_fair__${K}__tensorrt-llm__eqkv__c256_s256 tensorrtllm $M "{\"kv_cache_config\":{\"max_tokens\":$POOL}}"
echo QUEUE_DONE >> /home/ubuntu/projects/InferenceBench/results/l40s_fairness/_provenance/progress.log
