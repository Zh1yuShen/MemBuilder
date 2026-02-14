#!/bin/bash
# =============================================================================
# GRPO多机训练脚本 - verl + Ray 集群模式
# 
# 使用方式:
#   方式1: 手动启动 Ray 集群后，在 head 节点运行此脚本
#   方式2: 单机模式直接运行
#
# 注意: verl 使用 Ray 进行分布式调度，此脚本只需在 head 节点运行一次！
#
# 操作流程:
#   0. 查看IP: hostname -I | awk '{print $1}'
#   1. 主机(Head): NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=bond1 ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-gpus=8
#   2. 从机(Worker): NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=bond1 ray start --address=<主机IP>:6379 --num-gpus=8
#   3. 主机: ray status  # 确认2节点16GPU
#   4. 主机: ./run_memory_grpo_multinode.sh
#   5. 停止: Ctrl+C 或 ray stop --force
#   6. 清理: 两台机器都执行 ray stop --force
# =============================================================================

set -x

# ============================================
# 环境配置（不使用 export）
# ============================================
NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
RAY_ADDRESS=${RAY_ADDRESS:-auto}

NCCL_DEBUG=${NCCL_DEBUG:-INFO}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}

AUTO_IFNAME=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'dev \K\S+' | head -1)
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${AUTO_IFNAME:-eth0}}
NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5}

PROJECT_ROOT="$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/..")"
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ============================================
# 用户配置
# ============================================
# ⭐ Reward Server 地址
REWARD_SERVER_URL=${REWARD_SERVER_URL:-"http://localhost:8765"}

# 默认路径（可通过环境变量覆盖）
MODEL_PATH=${MODEL_PATH:-""}
TRAIN_DATA=${TRAIN_DATA:-"${PROJECT_ROOT}/data/memory_rl_train.parquet"}
DATASET_NAME=${DATASET_NAME:-"membuilder"}
RESUME_MODE=${RESUME_MODE:-"disable"}

OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/${DATASET_NAME}_${NNODES}nodes"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${PROJECT_ROOT}/checkpoints/${DATASET_NAME}"}

REWARD_FN_PATH="${PROJECT_ROOT}/training/reward_server/reward_function.py"
REWARD_CONFIG_PATH=${REWARD_CONFIG_PATH:-"${PROJECT_ROOT}/training/reward_server/reward_config.json"}

if [ -z "${MODEL_PATH}" ]; then
    echo "❌ ERROR: Please set MODEL_PATH to your SFT model path"
    echo "Usage: MODEL_PATH=/path/to/model ./scripts/run_memory_grpo_multinode.sh"
    exit 1
fi

if [ ! -f "${TRAIN_DATA}" ]; then
    echo "❌ ERROR: Training data not found at ${TRAIN_DATA}"
    echo "Run: python scripts/prepare_rl_data.py --trajectories-dir YOUR_TRAJECTORY_DIR --output-file ${TRAIN_DATA}"
    exit 1
fi

# Check if reward server is running
if ! curl -s "${REWARD_SERVER_URL}/health" > /dev/null 2>&1; then
    echo "⚠️  WARNING: Reward server not responding at ${REWARD_SERVER_URL}"
    echo "Start it with: cd training/reward_server && ./start_server.sh"
fi

echo "============================================"
echo "RL Training Configuration"
echo "============================================"
echo "Model: ${MODEL_PATH}"
echo "Data: ${TRAIN_DATA}"
echo "Reward Server: ${REWARD_SERVER_URL}"
echo "Nodes: ${NNODES} × ${GPUS_PER_NODE} GPUs"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/val_generations
mkdir -p ${CHECKPOINT_DIR}

if [ $NNODES -eq 1 ]; then
    REWARD_MAX_CONCURRENT=${REWARD_MAX_CONCURRENT:-4}
elif [ $NNODES -eq 2 ]; then
    REWARD_MAX_CONCURRENT=${REWARD_MAX_CONCURRENT:-8}
elif [ $NNODES -ge 4 ]; then
    REWARD_MAX_CONCURRENT=${REWARD_MAX_CONCURRENT:-16}
fi

BASE_BATCH_SIZE=${BASE_BATCH_SIZE:-32}
train_batch_size=$((BASE_BATCH_SIZE * NNODES))
ppo_mini_batch_size=${ppo_mini_batch_size:-${train_batch_size}}
ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu:-4}
log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu:-4}

tensor_model_parallel_size=${tensor_model_parallel_size:-1}
gpu_memory_utilization=${gpu_memory_utilization:-0.7}
n_resp_per_prompt=${n_resp_per_prompt:-8}

PYTHONPATH="${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
RAY_ADDRESS="${RAY_ADDRESS}" \
NCCL_DEBUG="${NCCL_DEBUG}" \
NCCL_IB_DISABLE="${NCCL_IB_DISABLE}" \
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX}" \
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
NCCL_IB_HCA="${NCCL_IB_HCA}" \
REWARD_MAX_CONCURRENT="${REWARD_MAX_CONCURRENT}" \
MEMBUILDER_MEMORY_API_URL="${REWARD_SERVER_URL}" \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${TRAIN_DATA} \
    data.train_batch_size=${train_batch_size} \
    data.shuffle=False \
    data.max_prompt_length=20000 \
    data.max_response_length=6000 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    +actor_rollout_ref.model.override_config.torch_dtype="bfloat16" \
    +actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.local_files_only=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.max_length=20000 \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    'actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_model]' \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype="bfloat16" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    \
    +actor_rollout_ref.ref.dtype="bfloat16" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='memory_grpo' \
    trainer.experiment_name="${DATASET_NAME}_${NNODES}nodes" \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=10 \
    trainer.default_local_dir=${CHECKPOINT_DIR} \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs=5 \
    +trainer.output_dir=${OUTPUT_DIR} \
    +trainer.validation_data_dir=${OUTPUT_DIR}/val_generations \
    \
    reward_model.reward_manager=batch \
    custom_reward_function.path=${REWARD_FN_PATH} \
    custom_reward_function.name=compute_score_batch
