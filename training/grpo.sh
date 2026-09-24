#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA:?Set DATA to the prepared grpo.jsonl}"
: "${SFT_ADAPTER:?Set SFT_ADAPTER to the final SFT adapter directory}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
[[ "$NPROC_PER_NODE" == 8 ]] || { echo 'The published GRPO schedule uses 8 GPUs.' >&2; exit 1; }
export USE_AUDIO_IN_VIDEO=true FPS=1.0 FPS_MIN_FRAMES=2
export USE_HF=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset FPS_MAX_FRAMES VIDEO_MAX_FRAMES MAX_NUM_FRAMES VIDEO_MAX_PIXELS
swift rlhf \
  --rlhf_type grpo --model "${MODEL:-Qwen/Qwen2.5-Omni-7B}" --model_type qwen2_5_omni \
  --dataset "$DATA" --adapters "$SFT_ADAPTER" --ref_adapters "$SFT_ADAPTER" \
  --external_plugins training/reward.py --reward_funcs timeduet --reward_weights 1.0 \
  --tuner_type lora --torch_dtype bfloat16 --freeze_vit true --freeze_aligner true \
  --lora_rank 8 --lora_alpha 32 --lora_dropout 0.05 --target_modules all-linear \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --num_generations 4 \
  --learning_rate 1e-5 --weight_decay 0.1 --lr_scheduler_type cosine --warmup_ratio 0.05 \
  --max_grad_norm 1.0 --beta 0.04 --epsilon 0.20 --max_length 32768 \
  --max_completion_length 512 --max_steps 300 --seed 20260801 --data_seed 42 \
  --temperature 1.0 --top_p 0.95 --save_steps 50 --logging_steps 1 \
  --output_dir "${OUTPUT:-outputs/grpo}" --use_vllm false --eval_strategy no \
  --gradient_checkpointing true --ddp_find_unused_parameters false \
  --dataset_shuffle true --train_dataloader_shuffle true --dataloader_drop_last false \
  --report_to none
