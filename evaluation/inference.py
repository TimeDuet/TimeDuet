#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

from dataset.prepare import read_split, prepare_row
from dataset.build import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]



def merge_intervals(intervals, duration):
    valid = []
    for item in intervals if isinstance(intervals, list) else []:
        if not isinstance(item, list) or len(item) != 2:
            continue
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        start = max(0.0, min(duration, start))
        end = max(0.0, min(duration, end))
        if end <= start:
            continue
        valid.append([start, end])
    valid.sort()
    merged = []
    for start, end in valid:
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def strict_parse(text, duration):
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    envelope_valid = len(matches) == 1 and re.fullmatch(
        r"\s*<answer>.*?</answer>\s*", text, flags=re.DOTALL
    ) is not None
    if not envelope_valid:
        return False, [], "answer_tag_envelope"
    try:
        payload = json.loads(matches[0].strip())
    except json.JSONDecodeError as exc:
        return False, [], f"json:{exc.msg}"
    if not isinstance(payload, dict) or set(payload) != {"intervals"}:
        return False, [], "schema"
    if not isinstance(payload["intervals"], list):
        return False, [], "intervals_not_list"
    for item in payload["intervals"]:
        if not isinstance(item, list) or len(item) != 2:
            return False, [], "interval_shape"
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in item):
            return False, [], "interval_non_numeric"
    return True, merge_intervals(payload["intervals"], duration), ""



def probe_video(path):
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of", "json", str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    stream = payload["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    fps = float(numerator) / float(denominator)
    total_frames = int(stream["nb_frames"])
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "source_fps": fps,
        "source_total_frames": total_frames,
        "source_duration": float(stream.get("duration") or total_frames / fps),
    }


def load_model(model_path, adapter_path):
    import torch
    from peft import PeftModel
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.disable_talker()
    if getattr(model, "has_talker", True):
        raise RuntimeError("talker disable failed")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return model, processor


def prompt_parts(row, policy):
    system = row["messages"][0]["content"]
    user = row["messages"][1]["content"]
    expected_user = policy["prompt"]["user_template"].format(query=row["query"])
    if system != policy["prompt"]["system"] or user != expected_user:
        raise ValueError(f"prompt mismatch for {row['query_id']}")
    if not user.startswith("<video>\n"):
        raise ValueError(f"missing video placeholder for {row['query_id']}")
    return system, user[len("<video>\n"):]


def evaluate_row(model, processor, row, raw_row, policy):
    import torch
    from qwen_omni_utils import process_mm_info
    from qwen_omni_utils.v2_5.vision_process import smart_nframes

    started = time.time()
    duration = float(row["duration_sec"])
    video_path = Path(row["videos"][0])
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    system, user_text = prompt_parts(row, policy)
    video_config = {
        "type": "video",
        "video": str(video_path),
        "fps": float(policy["input"]["video_sampling_fps"]),
    }
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [video_config, {"type": "text", "text": user_text}]},
    ]
    rendered_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    inputs = processor(
        text=rendered_prompt,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    )
    required = ("input_ids", "video_grid_thw", "input_features", "feature_attention_mask")
    missing = [key for key in required if key not in inputs]
    if missing:
        raise RuntimeError(f"missing multimodal tensors: {missing}")
    if inputs["input_features"].numel() == 0 or inputs["feature_attention_mask"].numel() == 0:
        raise RuntimeError("empty audio tensor")

    temporal_patch_size = int(getattr(processor.video_processor, "temporal_patch_size", 2))
    selected_frames = int(inputs["video_grid_thw"][0][0]) * temporal_patch_size
    source = probe_video(video_path)
    expected_frames = smart_nframes(
        video_config,
        total_frames=source["source_total_frames"],
        video_fps=source["source_fps"],
    )
    if selected_frames != expected_frames:
        raise RuntimeError(f"frame mismatch: selected={selected_frames}, expected={expected_frames}")
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    generated = model.generate(
        **inputs,
        use_audio_in_video=True,
        return_audio=False,
        do_sample=False,
        num_beams=1,
        max_new_tokens=int(policy["generation"]["max_new_tokens"]),
    )
    new_tokens = generated[:, inputs.input_ids.shape[1]:]
    generated_tokens = int(new_tokens.shape[1])
    max_new_tokens = int(policy["generation"]["max_new_tokens"])
    cap_hit = generated_tokens >= max_new_tokens
    response = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    parse_valid, prediction, parse_error = strict_parse(response, duration)
    if not parse_valid:
        prediction = []
    return {
        "query_id": row["query_id"],
        "sample_id": row["sample_id"],
        "query": row["query"],
        "temporal_relation": row["temporal_relation"],
        "shift_seconds": row["audio_shift_sec"],
        "shift_magnitude_seconds": abs(float(row["audio_shift_sec"])),
        "duration_seconds": duration,
        "num_slots": raw_row.get("num_slots"),
        "slot": raw_row.get("slot"),
        "video_path": str(video_path),
        "raw_response": response,
        "strict_parse_valid": parse_valid,
        "strict_parse_error": parse_error,
        "strict_prediction_intervals": prediction,
        "total_gt": raw_row["total_gt"],
        "audio_gt": raw_row["audio_gt"],
        "visual_gt": raw_row["visual_gt"],
        "selected_frame_count": selected_frames,
        "expected_frame_count": expected_frames,
        "video_grid_thw": inputs["video_grid_thw"].detach().cpu().tolist(),
        "encoded_input_length": int(inputs["input_ids"].shape[1]),
        "audio_tensor_nonempty": True,
        "input_features_shape": list(inputs["input_features"].shape),
        "feature_attention_mask_shape": list(inputs["feature_attention_mask"].shape),
        "source_video": source,
        "input_policy": video_config,
        "attention_implementation": "sdpa",
        "talker_disabled": True,
        "generated_tokens": generated_tokens,
        "termination_reason": "length_cap" if cap_hit else "model_native_eos_or_conversation_separator",
        "cap_hit": cap_hit,
        "provider_finish_reason": None,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "prompt_sha256": hashlib.sha256(rendered_prompt.encode()).hexdigest(),
        "latency_seconds": round(time.time() - started, 3),
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser(description="Run TimeDuet-Qwen on a TimeDuet split.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--adapter", default="TimeDuet/TimeDuet-Qwen",
                        help="LoRA adapter path or Hub ID; use an empty string for the base model.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for name in ("FPS_MAX_FRAMES", "VIDEO_MAX_FRAMES", "MAX_NUM_FRAMES", "VIDEO_MAX_PIXELS"):
        os.environ.pop(name, None)
    if not 0 <= args.shard_index < args.num_shards or args.limit < 0:
        parser.error("Invalid shard selection or limit")
    raw_rows = read_split(args.data, args.split)
    dataset_hash = hashlib.sha256(json.dumps(raw_rows, sort_keys=True).encode()).hexdigest()
    size, remainder = divmod(len(raw_rows), args.num_shards)
    start = args.shard_index * size + min(args.shard_index, remainder)
    raw_rows = raw_rows[start:start + size + int(args.shard_index < remainder)]
    if args.limit:
        raw_rows = raw_rows[:args.limit]
    if not raw_rows:
        parser.error("No queries selected")
    rows = [prepare_row(row, args.data) for row in raw_rows]
    policy = {
        "prompt": {"system": SYSTEM_PROMPT, "user_template": USER_PROMPT_TEMPLATE},
        "input": {"video_sampling_fps": 1.0},
        "generation": {"max_new_tokens": 1024, "do_sample": False, "num_beams": 1},
    }
    metadata = {
        "model": args.model, "adapter": args.adapter, "split": args.split,
        "dataset_sha256": dataset_hash, "policy": policy,
        "selected_query_ids": [row["query_id"] for row in rows],
    }
    metadata_path = args.out.with_suffix(".run.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if args.out.exists():
        if not args.resume:
            raise FileExistsError(f"{args.out} exists; use --resume or a new output path")
        if not metadata_path.is_file() or json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("Cannot resume: data, model, or evaluation settings differ")
        for record in read_jsonl(args.out):
            query_id = record["query_id"]
            if query_id in completed or query_id not in metadata["selected_query_ids"] or record.get("error"):
                raise ValueError("Invalid or duplicate record in resume file")
            completed[query_id] = record
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    pending = [(row, raw) for row, raw in zip(rows, raw_rows) if row["query_id"] not in completed]
    if pending:
        import torch
        model, processor = load_model(args.model, args.adapter)
        with args.out.open("a", encoding="utf-8") as handle, torch.inference_mode():
            for row, raw in pending:
                record = evaluate_row(model, processor, row, raw, policy)
                record["model_id"] = args.adapter or args.model
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                completed[row["query_id"]] = record
                print(f'{len(completed)}/{len(rows)} {row["query_id"]}', flush=True)
    print(f"Saved {len(completed)} predictions to {args.out}")


if __name__ == "__main__":
    main()
