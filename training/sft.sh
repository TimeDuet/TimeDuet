#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA:?Set DATA to the prepared sft.jsonl}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
[[ "$NPROC_PER_NODE" == 2 ]] || { echo 'The published SFT schedule uses 2 GPUs.' >&2; exit 1; }
export USE_AUDIO_IN_VIDEO=true ENABLE_AUDIO_OUTPUT=false
export USE_HF=1
export FPS=1.0 FPS_MIN_FRAMES=2 FPS_MAX_FRAMES=128 VIDEO_MAX_PIXELS=262144
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
swift sft \
  --model "${MODEL:-Qwen/Qwen2.5-Omni-7B}" --model_type qwen2_5_omni \
  --dataset "$DATA" --tuner_type lora --torch_dtype bfloat16 \
  --freeze_vit true --freeze_aligner true --lora_rank 8 --lora_alpha 32 \
  --lora_dropout 0.05 --target_modules all-linear --loss_scale last_round \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 64 \
  --learning_rate 1e-5 --weight_decay 0.0 --lr_scheduler_type cosine \
  --warmup_ratio 0.05 --max_grad_norm 1.0 --max_length 32768 \
  --truncation_strategy delete --num_train_epochs 3 \
  --gradient_checkpointing true --ddp_find_unused_parameters false \
  --packing false --dataloader_drop_last false --dataloader_num_workers 8 \
  --dataloader_prefetch_factor 2 --dataloader_persistent_workers true \
  --dataset_shuffle true --train_dataloader_shuffle true \
  --seed 20260720 --data_seed 20260720 --save_strategy steps --save_steps 50 \
  --eval_strategy no --logging_steps 1 --report_to none \
  --output_dir "${OUTPUT:-outputs/sft}"
