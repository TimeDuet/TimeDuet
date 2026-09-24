# TimeDuet

**A Benchmark for Audio-Visual Temporal Grounding on Misaligned Streams**

[Project](https://github.com/TimeDuet) · [Dataset](https://huggingface.co/datasets/TimeDuet/TimeDuet) · [Model](https://huggingface.co/TimeDuet/TimeDuet-Qwen)

Code for dataset construction, SFT and GRPO training, inference, and the TimeDuet evaluation protocol. Synthesized videos, annotations, and the final model are hosted on Hugging Face.

## Setup

Use Python 3.10 with a CUDA-capable PyTorch environment. Install FFmpeg and FFprobe on your system, then:

```bash
git clone https://github.com/TimeDuet/TimeDuet.git
cd TimeDuet
pip install -r requirements.txt
```

Run the commands below from the repository root. The pinned packages match the reference environment; GPU inference and training require sufficient device memory.

## Dataset

```bash
hf download TimeDuet/TimeDuet --repo-type dataset --local-dir data/TimeDuet
```

| Split | Videos | Queries |
|---|---:|---:|
| Train | 2,686 | 18,805 |
| Validation | 315 | 2,207 |
| Test | 315 | 2,202 |

Each query has separate `audio_gt` and `visual_gt` intervals. The model predicts their union, `total_gt`, in seconds. Keep the released splits unchanged. The download contains synthesized media; the separate source clips are not included.

## Inference and evaluation

```bash
python -m evaluation.inference \
  --data data/TimeDuet --split test \
  --out outputs/test.jsonl

python -m evaluation.metrics \
  --input outputs/test.jsonl \
  --output outputs/test-scored.jsonl \
  --summary outputs/test-metrics.json
```

Inference loads `TimeDuet/TimeDuet-Qwen` as a LoRA adapter on `Qwen/Qwen2.5-Omni-7B`. It uses embedded audio, video sampled at 1 FPS, greedy decoding, and a maximum of 1,024 new tokens. Add `--limit 2` with a different output path for a short run. To continue an interrupted run with the same settings, add `--resume`. For the base model, pass `--adapter ""`.

The scorer reports mIoU, R@0.3/0.5/0.7, A-IoU, V-IoU, AV Overlap, Audio-only, and Visual-only on a 0–1 scale. Audio-only and Visual-only are averaged over the 1,468 shifted test queries. A-IoU and V-IoU remove the opposite modality's exclusive region from the prediction before computing IoU. Invalid answer formats are scored as empty predictions; runtime failures must be resolved before scoring. If using `--num-shards` and `--shard-index`, combine all prediction shards before scoring the full split.

## Training

Prepare the training inputs from the released data:

```bash
python -m dataset.prepare --data data/TimeDuet --output data/training
```

SFT uses all 18,805 training queries. GRPO uses the 12,536 audio-leading or audio-delayed queries, with the same source order and prompt as the released benchmark.

```bash
# SFT: 2 GPUs, 3 epochs.
DATA=data/training/sft.jsonl OUTPUT=outputs/sft bash training/sft.sh

# Continue from the final SFT adapter; do not merge it into the base model.
SFT_ADAPTER="$(find outputs/sft -type d -name checkpoint-441 -print -quit)"
DATA=data/training/grpo.jsonl SFT_ADAPTER="$SFT_ADAPTER" \
  OUTPUT=outputs/grpo bash training/grpo.sh
```

Both stages use rank-8 LoRA with alpha 32 and learning rate 1e-5. SFT uses two GPUs with gradient accumulation 64; GRPO uses eight GPUs with gradient accumulation 8 and four generations per query. The scripts preserve these device counts to retain the published schedule. Select the intended devices with `CUDA_VISIBLE_DEVICES` when needed.

The GRPO reward combines overall IoU and the lower modality-specific IoU, with a penalty for their imbalance. Training follows a 300-step cosine schedule; the released model is the selected step-200 adapter. Shortening the schedule to 200 steps changes the learning-rate trajectory.

## Dataset construction

To synthesize videos from your locally available, selected source clips, provide a CSV with `clip_index`, `youtube_id`, `start_seconds`, `end_seconds`, `label`, and `clip_path`. Source times refer to the original video; `clip_path` points to the extracted clip and may be relative to the CSV. Preserve the selected source set and its row order when reproducing the dataset. Source IDs and segment provenance are available in the released sample manifests.

```bash
python -m dataset.build \
  --accepted-csv data/accepted_clips.csv \
  --out-root data/synthesized --exhaustive --workers 2
```

The default configuration uses 23,214 selected clips, seed 20260719, 5–9 target events per video, and the included label-conflict rules. Output includes videos and per-query/per-sample manifests. If the source clips have moved, use `--clips-dir` to override their directory. `--manifest-only` checks construction without rendering videos. For a different source collection, set `--expected-source-rows` to its size. Internal schema identifiers are retained for compatibility with the released annotations.

## Citation

Sihyeong Kim, Jaeyeong Choi, Joohyun Oh, Taeyeong Jeong, Bomin Kang, Daehee Park, and Jisoo Mok. *TimeDuet: A Benchmark for Audio-Visual Temporal Grounding on Misaligned Streams.*

Please also acknowledge the underlying datasets and models when using them. Their respective terms and attribution requirements still apply.
