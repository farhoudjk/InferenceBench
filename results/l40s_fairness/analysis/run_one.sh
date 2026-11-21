#!/usr/bin/env bash
# Usage: run_one.sh RUN_ID ENGINE MODEL [TRT_EXTRA_JSON|-] [EXTRA_ENGINE_ARG|-]
# One fresh server launch per rate (matches baseline harness behaviour); each
# rate's engine log and harness stdout are preserved under <run_dir>/logs/.
set -u
cd /home/ubuntu/projects/InferenceBench
RUN_ID=$1; ENGINE=$2; MODEL=$3; TRT_EXTRA=${4:--}; EXTRA=${5:--}
# eqkv: POOL = min of this model's Nov-2026 *default* pools (read from their launch logs),
# overriding any value passed by the queue script.
if [[ $RUN_ID == *__eqkv__* ]]; then
  MKEY=$(echo $RUN_ID | awk -F'__' '{print $2}')
  POOL=$(/home/ubuntu/projects/Bench2/.venv/bin/python results/l40s_fairness/analysis/kv_pools.py --pool $MKEY) || { echo "$RUN_ID: cannot compute POOL" >> results/l40s_fairness/_provenance/failures.log; exit 1; }
  echo "$RUN_ID POOL=$POOL" >> results/l40s_fairness/_provenance/pools_used.log
  case $ENGINE in
    vllm) EXTRA="--num-gpu-blocks-override=$((POOL/16))"; TRT_EXTRA=- ;;
    sglang) EXTRA="--max-total-tokens=$POOL"; TRT_EXTRA=- ;;
    tensorrtllm) TRT_EXTRA="{\"kv_cache_config\":{\"max_tokens\":$POOL}}"; EXTRA=- ;;
  esac
fi
PY=/home/ubuntu/projects/Bench2/.venv/bin/python
export PATH=/home/ubuntu/projects/Bench2/.venv/bin:$PATH
export INFERENCEBENCH_DATASET_CACHE=/home/ubuntu/projects/InferenceBench/results/l40s_fairness/_provenance/dataset_cache
OUT=results/l40s_fairness; RD=$OUT/$RUN_ID; mkdir -p $RD/logs
PROV=$OUT/_provenance/launches
for RATE in low high; do
  if [ -f $RD/logs/.done_$RATE ]; then continue; fi
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > $RD/logs/gpu_procs_before_$RATE.txt
  if [ -s $RD/logs/gpu_procs_before_$RATE.txt ]; then echo "GPU busy before $RUN_ID $RATE" >> $OUT/_provenance/failures.log; fi
  CMD=($PY main.py --model "$MODEL" --engine $ENGINE --results-dir $OUT --run-id $RUN_ID
       --strategy baseline_full_fp16_tp1 --workload mixed_real --num-requests 500
       --rates $RATE --arrival-patterns poisson --max-concurrent 256 --max-num-seqs-override 256
       --ttft-timeout 300 --request-timeout 600 --seed 42)
  [ "$EXTRA" != "-" ] && CMD+=("--extra-engine-args=$EXTRA")
  if [ "$TRT_EXTRA" != "-" ]; then export TRTLLM_EXTRA_LLM_API_OPTIONS="$TRT_EXTRA"; else unset TRTLLM_EXTRA_LLM_API_OPTIONS; fi
  { date -u +%FT%TZ; echo "TRTLLM_EXTRA_LLM_API_OPTIONS=${TRTLLM_EXTRA_LLM_API_OPTIONS:-}"; printf '%q ' "${CMD[@]}"; echo; } > $PROV/${RUN_ID}__$RATE.cmd
  "${CMD[@]}" > $RD/logs/harness_$RATE.log 2>&1; RC=$?
  for f in $RD/*.log; do [ -f "$f" ] && mv "$f" $RD/logs/$(basename ${f%.log})__$RATE.log; done
  for f in $RD/trtllm_config_*.yaml; do [ -f "$f" ] && cp "$f" $RD/logs/$(basename ${f%.yaml})__$RATE.yaml; done
  echo "$RUN_ID $RATE rc=$RC $(date -u +%FT%TZ)" >> $OUT/_provenance/progress.log
  if grep -q "Failed \[" $RD/logs/harness_$RATE.log; then RC=99; fi
  [ $RC -eq 0 ] && touch $RD/logs/.done_$RATE || echo "$RUN_ID $RATE rc=$RC" >> $OUT/_provenance/failures.log
done
