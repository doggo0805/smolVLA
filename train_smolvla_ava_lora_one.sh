#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace/openvla-oft/train_logs /workspace/openvla-oft/checkpoints

RUN_NAME="train_smolvla_ava_lora_one"
CKPT="/workspace/openvla-oft/checkpoints/${RUN_NAME}"
LOG="/workspace/openvla-oft/train_logs/${RUN_NAME}.log"
PID_FILE="/workspace/openvla-oft/train_logs/train_smolvla_ava_lora_one_pid.txt"
LAUNCH_INFO="/workspace/openvla-oft/train_logs/train_smolvla_ava_lora_one_launch.txt"

printf 'RUN_NAME=%s\nCKPT=%s\nLOG=%s\n' "${RUN_NAME}" "${CKPT}" "${LOG}" > "${LAUNCH_INFO}"

CONDA_ENV="smolvlaWu"
CONDA_BIN="/opt/conda/bin/conda"
if [[ ! -x "${CONDA_BIN}" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
fi

(
  cd /workspace/openvla-oft/lerobot
  export CUDA_VISIBLE_DEVICES=0

  exec "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    lerobot-train \
      --seed=1000 \
      --policy.type=smolvla \
      --policy.load_vlm_weights=true \
      --policy.freeze_vision_encoder=false \
      --policy.train_expert_only=false \
      --policy.train_state_proj=true \
      --policy.resize_imgs_with_padding='[512,512]' \
      --policy.attention_mode=cross_attn \
      --policy.self_attn_every_n_layers=2 \
      --policy.num_vlm_layers=16 \
      --policy.num_expert_layers=-1 \
      --policy.expert_width_multiplier=0.75 \
      --policy.chunk_size=10 \
      --policy.n_action_steps=10 \
      --policy.num_steps=10 \
      --policy.optimizer_lr=1e-4 \
      --policy.optimizer_betas='[0.9,0.95]' \
      --policy.optimizer_weight_decay=1e-10 \
      --policy.optimizer_grad_clip_norm=5 \
      --policy.scheduler_warmup_steps=1000 \
      --policy.scheduler_decay_steps=30000 \
      --policy.scheduler_decay_lr=2.5e-6 \
      --dataset.repo_id=/workspace/openvla-oft/data/libero_all \
      --batch_size=16 \
      --steps=100000 \
      --eval_freq=0 \
      --save_checkpoint=true \
      --save_freq=20000 \
      --output_dir="${CKPT}" \
      --policy.device=cuda \
      --policy.push_to_hub=false \
      --policy.use_peft=true \
      --policy.r=32 \
      --policy.lora_alpha=64 \
      --policy.use_ava=true \
      --policy.ava_chunk_len=10 \
      --policy.ava_action_dim=7 \
      --policy.ava_action_tokens_len=70 \
      --policy.ava_hidden_dim=512 \
      --policy.ava_layer_idx=15 \
      --policy.ava_score_config='[1.9,0.1,0.0]' \
      --policy.ava_lambda_reg=1.0 \
      --policy.ava_reg_target_c=0.6 \
      --policy.ava_tbptt_steps=4 \
      --policy.ava_detach_step=2
) > "${LOG}" 2>&1 &

PID="$!"
echo "${PID}" > "${PID_FILE}"

echo "Started SmolVLA Libero-All training!"
echo "PID=${PID}"
echo "CKPT=${CKPT}"
echo "LOG=${LOG}"
