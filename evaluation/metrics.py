#!/usr/bin/env python3
"""Recompute TimeDuet evidence-region metrics from parsed interval predictions.

Evidence-region definitions:

  T-IoU = IoU(P, A union V)
  A-IoU = IoU(P minus (V minus A), A)
  V-IoU = IoU(P minus (A minus V), V)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

EPS = 1e-9


def normalize(intervals):
    cleaned = []
    for interval in intervals or []:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            continue
        start, end = float(interval[0]), float(interval[1])
        if end > start + EPS:
            cleaned.append((start, end))
    cleaned.sort()
    merged = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1] + EPS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def duration(intervals):
    return sum(end - start for start, end in intervals)


def intersection_duration(left, right):
    total = 0.0
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            total += end - start
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def intersection(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start + EPS:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return normalize(result)


def subtract(base, remove):
    result = []
    for start, end in base:
        cursor = start
        for remove_start, remove_end in remove:
            if remove_end <= cursor + EPS:
                continue
            if remove_start >= end - EPS:
                break
            if remove_start > cursor + EPS:
                result.append((cursor, min(remove_start, end)))
            cursor = max(cursor, remove_end)
            if cursor >= end - EPS:
                break
        if cursor < end - EPS:
            result.append((cursor, end))
    return normalize(result)


def iou(prediction, gold):
    pred_length = duration(prediction)
    gold_length = duration(gold)
    union = pred_length + gold_length - intersection_duration(prediction, gold)
    return 1.0 if union <= EPS and pred_length <= EPS and gold_length <= EPS else (intersection_duration(prediction, gold) / union if union > EPS else 0.0)



def score_row(row):
    if row.get("error"):
        raise ValueError(f"Inference error: {row.get('query_id')}")
    for key in ("query_id", "audio_gt", "visual_gt", "shift_seconds",
                "strict_parse_valid", "strict_prediction_intervals"):
        if key not in row:
            raise ValueError(f"Missing prediction field: {key}")
    if not row["audio_gt"] or not row["visual_gt"]:
        raise ValueError("Both modality ground truths must be nonempty")
    prediction = normalize(row.get("strict_prediction_intervals", [])) if row.get("strict_parse_valid") else []
    audio = normalize(row.get("audio_gt", []))
    visual = normalize(row.get("visual_gt", []))
    total = normalize(audio + visual)
    audio_only = subtract(audio, visual)
    visual_only = subtract(visual, audio)
    av_overlap = intersection(audio, visual)
    audio_prediction = subtract(prediction, visual_only)
    visual_prediction = subtract(prediction, audio_only)
    overlap_prediction = subtract(prediction, normalize(audio_only + visual_only))
    audio_only_prediction = subtract(prediction, visual)
    visual_only_prediction = subtract(prediction, audio)

    row["metric_version"] = "timeduet.evidence-region.v1"
    row["canonical_t_iou"] = iou(prediction, total)
    row["canonical_a_iou"] = iou(audio_prediction, audio)
    row["canonical_v_iou"] = iou(visual_prediction, visual)
    row["canonical_av_overlap_iou"] = iou(overlap_prediction, av_overlap) if duration(av_overlap) > EPS else None
    shifted = abs(float(row.get("shift_seconds", 0.0) or 0.0)) > EPS
    row["canonical_audio_only_iou"] = iou(audio_only_prediction, audio_only) if shifted and duration(audio_only) > EPS else None
    row["canonical_visual_only_iou"] = iou(visual_only_prediction, visual_only) if shifted and duration(visual_only) > EPS else None
    return row


def mean(rows, key):
    return fmean(row[key] for row in rows) if rows else 0.0


def eligible_mean(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return fmean(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    rows = []
    with Path(args.input).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(score_row(json.loads(line)))

    if not rows or len({r["query_id"] for r in rows}) != len(rows):
        raise ValueError("Empty predictions or duplicate query IDs")
    for path in (args.output, args.summary):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    tiou = [row["canonical_t_iou"] for row in rows]
    summary = {
        "metric_version": "timeduet.evidence-region.v1",
        "n": len(rows),
        "mIoU": mean(rows, "canonical_t_iou"),
        "A-IoU": mean(rows, "canonical_a_iou"),
        "V-IoU": mean(rows, "canonical_v_iou"),
        "AV Overlap": eligible_mean(rows, "canonical_av_overlap_iou"),
        "Audio-only": eligible_mean(rows, "canonical_audio_only_iou"),
        "Visual-only": eligible_mean(rows, "canonical_visual_only_iou"),
        "AV Overlap n": sum(row.get("canonical_av_overlap_iou") is not None for row in rows),
        "Audio-only n": sum(row.get("canonical_audio_only_iou") is not None for row in rows),
        "Visual-only n": sum(row.get("canonical_visual_only_iou") is not None for row in rows),
        "R@0.3": fmean(1.0 if value >= 0.3 else 0.0 for value in tiou) if tiou else 0.0,
        "R@0.5": fmean(1.0 if value >= 0.5 else 0.0 for value in tiou) if tiou else 0.0,
        "R@0.7": fmean(1.0 if value >= 0.7 else 0.0 for value in tiou) if tiou else 0.0,
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
