#!/bin/bash
# run_disaggregated_grpo.sh
# Vanilla Disaggregated Sync GRPO Job for Slime (Strategy 2: AutoBridge - 10 RL Loops)

set -ex

export PYTHONUNBUFFERED=1

MODEL_NAME="Qwen2.5-0.5B-Instruct"
LOCAL_MODEL_DIR="/tmp/${MODEL_NAME}"
LOCAL_PROMPT_DATA="/tmp/dapo-math-17k/dapo-math-17k.jsonl"
SAVE_DIR="${SAVE_DIR:-/tmp/slime_checkpoints}"

# 1. Download HuggingFace weights and dataset locally across ALL Ray nodes via hf_hub_download
echo "Ensuring model and dataset exist across all Ray nodes..."
python3 -c "
import ray, os
from huggingface_hub import snapshot_download, hf_hub_download

ray.init(address='auto', ignore_reinit_error=True)

@ray.remote
def download_assets():
    if not os.path.exists('${LOCAL_MODEL_DIR}'):
        print('Downloading ${MODEL_NAME} to ${LOCAL_MODEL_DIR}...')
        snapshot_download('Qwen/${MODEL_NAME}', local_dir='${LOCAL_MODEL_DIR}')
    if not os.path.exists('${LOCAL_PROMPT_DATA}'):
        print('Downloading dataset to ${LOCAL_PROMPT_DATA}...')
        hf_hub_download(repo_id='zhuzilin/dapo-math-17k', filename='dapo-math-17k.jsonl', repo_type='dataset', local_dir='/tmp/dapo-math-17k')
    return True

nodes = ray.nodes()
futures = [download_assets.options(scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(node_id=n['NodeID'], soft=False)).remote() for n in nodes if n['Alive']]
ray.get(futures)
print('Asset setup complete across all nodes.')
"

# 2. Qwen2.5-0.5B Megatron Architecture & AutoBridge Checkpoint Flags
MODEL_ARGS=(
    --hf-checkpoint ${LOCAL_MODEL_DIR}
    --ref-load ${LOCAL_MODEL_DIR}
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model Qwen/Qwen2.5-0.5B-Instruct
    --megatron-to-hf-mode bridge
    --save ${SAVE_DIR}
    --save-interval 5
    --swiglu
    --num-layers 24
    --hidden-size 896
    --ffn-hidden-size 4864
    --num-attention-heads 14
    --use-rotary-position-embeddings
    --disable-bias-linear
    --add-qkv-bias
    --normalization "RMSNorm"
    --norm-epsilon 1e-6
    --rotary-base 1000000
    --group-query-attention
    --num-query-groups 2
    --vocab-size 151936
)

# 3. Disaggregated Resource Allocation Config
RESOURCE_ARGS=(
    --actor-num-nodes 1
    --actor-num-gpus-per-node 1
    --rollout-num-gpus 1
    --rollout-num-gpus-per-engine 1
)

# 4. Rollout & Dataset Config (Configured for 10 RL loops)
ROLLOUT_ARGS=(
    --prompt-data ${LOCAL_PROMPT_DATA}
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type deepscaler
    --num-rollout 10
    --rollout-batch-size 2
    --n-samples-per-prompt 2
    --num-steps-per-rollout 1
    --global-batch-size 4
    --rollout-max-response-len 256
    --rollout-temperature 0.8
)

# 5. Megatron Parallelism & Optimization
PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 2048
)

# 6. GRPO & Optimizer Config
GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.01
    --eps-clip 0.2
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
)

SGLANG_ARGS=(
    --sglang-mem-fraction-static 0.4
)

TIMESLICE_ARGS=(
    --enable-timeslice
    --timeslice-orchestrator-addr "timeslice-acceleratororchestrator.timeslice-system.svc.cluster.local:50051"
    --timeslice-sampler-group "group-slime-sampler"
    --timeslice-trainer-group "group-slime-trainer"
    --timeslice-job-id "${TIMESLICE_JOB_ID:-slime-job-$$}"
)

# Launch Slime training via Ray
python3 /root/slime/train.py \
    "${RESOURCE_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${TIMESLICE_ARGS[@]}"
