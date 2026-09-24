#!/usr/bin/env python3
import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_BLOCKS = Path(__file__).with_name("label_blocks.csv")
DEFAULT_FFMPEG = Path(shutil.which("ffmpeg") or "ffmpeg")
EXPECTED_ACCEPTED_ROWS = 23214
EXPECTED_ACCEPTED_SHA256 = ""
DATASET_VERSION = "2026-07-20-mimo-v11-final-r7"
SAMPLE_SCHEMA_VERSION = "avrtg.sample.v1"
QUERY_SCHEMA_VERSION = "avrtg.query.v1"
PROMPT_POLICY_ID = "avrtg.temporal_grounding.union.v1"
SYSTEM_PROMPT = "You are an audio-visual temporal grounding model."
USER_PROMPT_TEMPLATE = """<video>
Localize every time interval where the following event is present
in the audio stream, the visual stream, or both:

{query}

Return JSON inside <answer> tags. Use seconds relative to t=0 of the full video.
Return all matching intervals in chronological order, merging overlapping or contiguous intervals.

Required output format:
<answer>{{"intervals":[[start_second,end_second], ...]}}</answer>"""
SAMPLE_CONDITION = "mixed_temporal_shift"
SLOT_RELATIONS = ["aligned", "audio_delay", "audio_lead"]


def run(cmd):
    subprocess.run(cmd, check=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_path(path):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"accepted source CSV not found: {resolved}")
    return resolved


def read_rows(path, clips_dir=None):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("clip_path") or not r.get("label"):
                continue
            p = Path(r["clip_path"])
            if clips_dir is not None:
                p = Path(clips_dir) / p.name
            if not p.is_absolute():
                p = Path(path).resolve().parent / p
            r["clip_path"] = str(p.resolve())
            if not p.exists():
                continue
            r["clip_index"] = int(r["clip_index"])
            rows.append(r)
    return rows


def read_block_pairs(path):
    pairs = set()
    if not path or not Path(path).exists():
        return pairs
    with Path(path).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a = (r.get("parent_label") or "").strip()
            b = (r.get("child_label") or "").strip()
            if a and b:
                pairs.add(tuple(sorted((a, b))))
    return pairs


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def interval_union(items):
    xs = sorted((float(a), float(b)) for a, b in items if float(b) > float(a))
    out = []
    for a, b in xs:
        if not out or a > out[-1][1]:
            out.append([a, b])
        else:
            out[-1][1] = max(out[-1][1], b)
    return [[round(a, 3), round(b, 3)] for a, b in out]


def interval_intersection(a, b):
    out = []
    for s1, e1 in a:
        for s2, e2 in b:
            s, e = max(s1, s2), min(e1, e2)
            if e > s:
                out.append([round(s, 3), round(e, 3)])
    return out


def accepted_window_duration(row):
    if row.get("start_seconds") not in (None, "") and row.get("end_seconds") not in (None, ""):
        duration = float(row["end_seconds"]) - float(row["start_seconds"])
        if duration > 0:
            return duration
    return 10.0


def source_segments_for_duration(window_duration, virtual_start, duration):
    window_duration = float(window_duration)
    virtual_start = float(virtual_start)
    remaining = float(duration)
    if window_duration <= 0:
        raise ValueError("source window duration must be positive")
    segments = []
    source_start = virtual_start % window_duration
    while remaining > 1e-9:
        available = window_duration - source_start
        take = min(remaining, available)
        if take <= 1e-9:
            source_start = 0.0
            continue
        segments.append({
            "start_sec": round(source_start, 6),
            "duration_sec": round(take, 6),
            "end_sec": round(source_start + take, 6),
        })
        remaining -= take
        source_start = 0.0
    return segments


def compact_clip(
    row,
    duration=None,
    source_start_sec=0.0,
    source_segments=None,
    connected_chain_offset_sec=None,
):
    out = {
        "clip_index": row["clip_index"],
        "youtube_id": row.get("youtube_id"),
        "label": row["label"],
        "source_split": row.get("source_split") or row.get("split"),
        "clip_file": row.get("clip_file"),
        "clip_path": row["clip_path"],
        "source_video_path": row.get("source_video_path"),
        "accepted_source_interval_sec": [
            float(row["start_seconds"]) if row.get("start_seconds") not in (None, "") else None,
            float(row["end_seconds"]) if row.get("end_seconds") not in (None, "") else None,
        ],
        "accepted_window_duration_sec": round(accepted_window_duration(row), 3),
    }
    if duration is not None:
        duration = float(duration)
        out["duration"] = round(duration, 3)
        if source_segments is None:
            source_segments = source_segments_for_duration(
                accepted_window_duration(row),
                source_start_sec,
                duration,
            )
        out["source_segments"] = source_segments
        out["source_crop_start_sec"] = round(
            float(source_segments[0]["start_sec"]),
            3,
        )
        out["source_crop_end_sec"] = round(
            float(source_segments[-1]["end_sec"]),
            3,
        )
        out["source_wrap_count"] = max(0, len(source_segments) - 1)
        if connected_chain_offset_sec is not None:
            out["connected_chain_offset_sec"] = round(
                float(connected_chain_offset_sec),
                3,
            )
    return out


def source_intervals_overlap(left, right):
    if left.get("youtube_id") != right.get("youtube_id"):
        return False
    left_start = float(left["start_seconds"])
    left_end = float(left["end_seconds"])
    right_start = float(right["start_seconds"])
    right_end = float(right["end_seconds"])
    return min(left_end, right_end) > max(left_start, right_start)


def temporal_overlap_components(rows):
    by_source = defaultdict(list)
    for row in rows:
        by_source[row.get("youtube_id")].append(row)

    components = []
    for source_rows in by_source.values():
        parent = list(range(len(source_rows)))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left, right):
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for left in range(len(source_rows)):
            for right in range(left + 1, len(source_rows)):
                if source_intervals_overlap(source_rows[left], source_rows[right]):
                    union(left, right)

        grouped = defaultdict(list)
        for index, row in enumerate(source_rows):
            grouped[find(index)].append(row)
        components.extend(grouped.values())
    return components


def find_cross_split_temporal_overlaps(splits):
    by_source = defaultdict(list)
    for split, rows in splits.items():
        for row in rows:
            by_source[row.get("youtube_id")].append((split, row))

    overlaps = []
    for youtube_id, items in by_source.items():
        for left in range(len(items)):
            left_split, left_row = items[left]
            for right in range(left + 1, len(items)):
                right_split, right_row = items[right]
                if left_split == right_split:
                    continue
                if source_intervals_overlap(left_row, right_row):
                    overlaps.append({
                        "youtube_id": youtube_id,
                        "left_split": left_split,
                        "right_split": right_split,
                        "left_clip_index": left_row["clip_index"],
                        "right_clip_index": right_row["clip_index"],
                        "left_label": left_row["label"],
                        "right_label": right_row["label"],
                        "left_interval": [
                            float(left_row["start_seconds"]),
                            float(left_row["end_seconds"]),
                        ],
                        "right_interval": [
                            float(right_row["start_seconds"]),
                            float(right_row["end_seconds"]),
                        ],
                    })
    return overlaps


def split_rows_by_label(rows, seed):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    splits = {"train": [], "val": [], "test": []}
    split_counts = {}
    repair_report = []
    for label, xs in by_label.items():
        xs = xs[:]
        rng.shuffle(xs)
        n = len(xs)
        if n < 3:
            counts = {"train": n, "val": 0, "test": 0}
        elif n < 5:
            counts = {"train": n - 1, "val": 1, "test": 0}
        else:
            n_val = max(1, math.floor(0.1 * n))
            n_test = max(1, math.floor(0.1 * n))
            counts = {"train": n - n_val - n_test, "val": n_val, "test": n_test}
        assignments = {}
        offset = 0
        for split in ["train", "val", "test"]:
            part = xs[offset:offset + counts[split]]
            for row in part:
                assignments[row["clip_index"]] = split
            offset += counts[split]

        components = temporal_overlap_components(xs)
        multi_component_indices = {
            row["clip_index"]
            for component in components
            if len(component) > 1
            for row in component
        }
        cross_split_components = [
            component
            for component in components
            if len({assignments[row["clip_index"]] for row in component}) > 1
        ]
        moved_to_train = Counter()
        label_repair = {
            "label": label,
            "components": [],
            "swaps": [],
        }
        for component in cross_split_components:
            original_splits = {
                row["clip_index"]: assignments[row["clip_index"]]
                for row in component
            }
            for row in component:
                original_split = assignments[row["clip_index"]]
                if original_split != "train":
                    moved_to_train[original_split] += 1
                assignments[row["clip_index"]] = "train"
            label_repair["components"].append({
                "youtube_id": component[0].get("youtube_id"),
                "clip_indices": sorted(row["clip_index"] for row in component),
                "original_splits": original_splits,
                "assigned_split": "train",
            })

        swap_candidates = [
            row for row in xs
            if assignments[row["clip_index"]] == "train"
            and row["clip_index"] not in multi_component_indices
        ]
        repair_rng = random.Random(f"{seed}:temporal_overlap_repair:{label}")
        repair_rng.shuffle(swap_candidates)
        for destination in ["val", "test"]:
            for _ in range(moved_to_train[destination]):
                if not swap_candidates:
                    raise RuntimeError(
                        f"cannot preserve split counts after overlap repair for label={label}"
                    )
                replacement = swap_candidates.pop()
                assignments[replacement["clip_index"]] = destination
                label_repair["swaps"].append({
                    "clip_index": replacement["clip_index"],
                    "from": "train",
                    "to": destination,
                })
        if cross_split_components:
            repair_report.append(label_repair)

        actual_counts = Counter(assignments.values())
        if any(actual_counts[split] != counts[split] for split in counts):
            raise RuntimeError(
                f"split count mismatch after overlap repair for label={label}: "
                f"expected={counts}, actual={dict(actual_counts)}"
            )

        for row in xs:
            row = dict(row)
            row["source_split"] = assignments[row["clip_index"]]
            splits[row["source_split"]].append(row)
        split_counts[label] = {"total": n, **counts}

    for split in splits:
        rng.shuffle(splits[split])
    return splits, split_counts, repair_report


def conflicts_with_existing(label, used_labels, block_pairs):
    if label in used_labels:
        return True
    return any(tuple(sorted((label, other))) in block_pairs for other in used_labels)


def choose_row(by_label, used_labels, block_pairs, rng):
    labels = [label for label, pool in by_label.items() if pool]
    rng.shuffle(labels)
    labels.sort(key=lambda label: len(by_label[label]), reverse=True)
    for label in labels:
        if conflicts_with_existing(label, used_labels, block_pairs):
            continue
        pool = by_label[label]
        if pool:
            return pool.pop()
    return None


def choose_filler_row(rows, used_labels, block_pairs, rng):
    candidates = rows[:]
    rng.shuffle(candidates)
    for row in candidates:
        if not conflicts_with_existing(row["label"], used_labels, block_pairs):
            return row
    return None


def make_exhaustive_slot_plan(n_items, min_slots, max_slots, rng):
    if n_items <= 0:
        return []
    if n_items < min_slots:
        return [n_items]
    sizes = list(range(min_slots, max_slots + 1))
    base = max(0, n_items // sum(sizes))
    lo = max(0, base - 30)
    hi = base + 31
    best = None
    for prefix in itertools.product(range(lo, hi + 1), repeat=len(sizes) - 1):
        used = sum(size * count for size, count in zip(sizes[:-1], prefix))
        rem = n_items - used
        if rem < 0 or rem % sizes[-1] != 0:
            continue
        last = rem // sizes[-1]
        if not (lo <= last <= hi):
            continue
        counts = list(prefix) + [last]
        if sum(counts) == 0:
            continue
        imbalance = max(counts) - min(counts)
        mean_slots = sum(size * count for size, count in zip(sizes, counts)) / sum(counts)
        score = (imbalance, abs(mean_slots - ((min_slots + max_slots) / 2)), sum(counts))
        if best is None or score < best[0]:
            best = (score, counts)
    if best is None:
        raise RuntimeError(f"cannot make balanced slot plan for {n_items} items")
    plan = []
    for size, count in zip(sizes, best[1]):
        plan.extend([size] * count)
    rng.shuffle(plan)
    return plan


def make_relation_quota(n_slots):
    base = n_slots // len(SLOT_RELATIONS)
    quota = {rel: base for rel in SLOT_RELATIONS}
    for rel in SLOT_RELATIONS[: n_slots % len(SLOT_RELATIONS)]:
        quota[rel] += 1
    return quota


def draw_slot_relation_sequence(n, relation_quota, rng):
    seq = []
    if n >= len(SLOT_RELATIONS):
        rels = SLOT_RELATIONS[:]
        rng.shuffle(rels)
        for rel in rels:
            if relation_quota.get(rel, 0) > 0 and len(seq) < n:
                seq.append(rel)
                relation_quota[rel] -= 1
    while len(seq) < n:
        available = [rel for rel, count in relation_quota.items() if count > 0]
        if not available:
            raise RuntimeError("relation quota exhausted before slot plan")
        rng.shuffle(available)
        rel = max(available, key=lambda item: relation_quota[item])
        seq.append(rel)
        relation_quota[rel] -= 1
    rng.shuffle(seq)
    return seq


def sample_slot_relation_sequence(n):
    seq = []
    while len(seq) < n:
        seq.extend(SLOT_RELATIONS)
    return seq[:n]


def flatten_by_label(by_label):
    return [row for xs in by_label.values() for row in xs]


def remaining_clip_count(by_label):
    return sum(len(xs) for xs in by_label.values())


def remaining_label_count(by_label):
    return sum(1 for xs in by_label.values() if xs)



def build_split(split_name, rows, block_pairs, args, samples_per_split):
    rng = random.Random(f"{args.seed}:{split_name}")
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
    filler_rows = rows[:]
    samples, queries = [], []

    if samples_per_split is None:
        slot_plan = make_exhaustive_slot_plan(
            remaining_clip_count(by_label),
            args.min_slots,
            args.max_slots,
            rng,
        )
        relation_quota = make_relation_quota(sum(slot_plan))
    else:
        slot_plan = [None] * samples_per_split
        relation_quota = None

    for sid_idx, planned_slots in enumerate(slot_plan):
        sample = None
        for _ in range(args.max_attempts):
            trial_by_label = {label: label_rows[:] for label, label_rows in by_label.items()}
            trial_relation_quota = relation_quota.copy() if relation_quota is not None else None
            n_slots = planned_slots if planned_slots is not None else rng.randint(args.min_slots, args.max_slots)
            if relation_quota is None:
                slot_relations = sample_slot_relation_sequence(n_slots)
                rng.shuffle(slot_relations)
            else:
                slot_relations = draw_slot_relation_sequence(
                    n_slots,
                    trial_relation_quota,
                    rng,
                )

            used_labels = set()
            gap_rows = []
            for _gap_idx in range(n_slots + 1):
                gap_row = choose_filler_row(filler_rows, used_labels, block_pairs, rng)
                if gap_row is None:
                    break
                used_labels.add(gap_row["label"])
                gap_rows.append(gap_row)
            if len(gap_rows) != n_slots + 1:
                continue

            slot_specs = []
            for slot_idx in range(n_slots):
                clip = choose_row(trial_by_label, used_labels, block_pairs, rng)
                if clip is None:
                    break
                used_labels.add(clip["label"])

                accepted_duration = accepted_window_duration(clip)
                max_clip_sec = min(args.max_clip_sec, accepted_duration)
                if max_clip_sec < args.min_clip_sec:
                    break
                clip_len = rng.uniform(args.min_clip_sec, max_clip_sec)
                crop_slack = max(0.0, accepted_duration - clip_len)
                crop_start = rng.uniform(0.0, crop_slack) if crop_slack > 1e-9 else 0.0
                relation = slot_relations[slot_idx]
                if relation == "aligned":
                    shift = 0.0
                else:
                    max_shift = min(
                        args.max_shift_frac * clip_len,
                        clip_len - args.min_overlap_sec,
                    )
                    if max_shift < args.min_shift_sec:
                        break
                    magnitude = rng.uniform(args.min_shift_sec, max_shift)
                    shift = magnitude if relation == "audio_delay" else -magnitude
                slot_specs.append({
                    "clip": clip,
                    "clip_len": clip_len,
                    "crop_start": crop_start,
                    "relation": relation,
                    "shift": shift,
                    "pad": abs(shift),
                })
            if len(slot_specs) != n_slots:
                continue

            gap_specs = []
            for gap_idx, gap_row in enumerate(gap_rows):
                left_extension = slot_specs[gap_idx - 1]["pad"] if gap_idx > 0 else 0.0
                right_extension = slot_specs[gap_idx]["pad"] if gap_idx < n_slots else 0.0
                chain_duration = (
                    left_extension
                    + args.aligned_filler_gap_sec
                    + right_extension
                )
                accepted_duration = accepted_window_duration(gap_row)
                if chain_duration <= accepted_duration:
                    chain_slack = accepted_duration - chain_duration
                    chain_start = (
                        rng.uniform(0.0, chain_slack)
                        if chain_slack > 1e-9
                        else 0.0
                    )
                else:
                    chain_start = 0.0
                gap_specs.append({
                    "row": gap_row,
                    "left_extension": left_extension,
                    "right_extension": right_extension,
                    "chain_duration": chain_duration,
                    "chain_start": chain_start,
                    "chain_source_segments": source_segments_for_duration(
                        accepted_duration,
                        chain_start,
                        chain_duration,
                    ),
                })

            total_duration = (
                sum(spec["clip_len"] + spec["pad"] for spec in slot_specs)
                + (n_slots + 1) * args.aligned_filler_gap_sec
            )
            if total_duration > args.max_total_sec:
                continue

            slots = []
            part_plan = []
            aligned_filler_gaps = []
            cursor = 0.0

            def add_aligned_gap(gap_idx, position, after_slot=None):
                nonlocal cursor
                gap_spec = gap_specs[gap_idx]
                gap_row = gap_spec["row"]
                start = cursor
                end = cursor + args.aligned_filler_gap_sec
                gap_clip = compact_clip(
                    gap_row,
                    args.aligned_filler_gap_sec,
                    source_segments=source_segments_for_duration(
                        accepted_window_duration(gap_row),
                        gap_spec["chain_start"] + gap_spec["left_extension"],
                        args.aligned_filler_gap_sec,
                    ),
                    connected_chain_offset_sec=gap_spec["left_extension"],
                )
                aligned_filler_gaps.append({
                    "position": position,
                    "after_slot": after_slot,
                    "interval": [round(start, 3), round(end, 3)],
                    "duration": round(args.aligned_filler_gap_sec, 3),
                    "filler_label": gap_row["label"],
                    "filler_clip": gap_clip,
                    "connected_source_chain": {
                        "source_virtual_start_sec": round(gap_spec["chain_start"], 3),
                        "left_shift_extension_sec": round(gap_spec["left_extension"], 3),
                        "aligned_gap_sec": round(args.aligned_filler_gap_sec, 3),
                        "right_shift_extension_sec": round(gap_spec["right_extension"], 3),
                        "chain_duration_sec": round(gap_spec["chain_duration"], 3),
                        "source_segments": gap_spec["chain_source_segments"],
                        "wrap_count": max(
                            0,
                            len(gap_spec["chain_source_segments"]) - 1,
                        ),
                    },
                })
                part_plan.append({
                    "kind": "aligned_filler_gap",
                    "position": position,
                    "after_slot": after_slot,
                    "clip": gap_clip,
                    "duration": round(args.aligned_filler_gap_sec, 3),
                })
                cursor = end

            add_aligned_gap(0, "start")
            for slot_idx, slot_spec in enumerate(slot_specs):
                clip = slot_spec["clip"]
                clip_len = slot_spec["clip_len"]
                crop_start = slot_spec["crop_start"]
                relation = slot_spec["relation"]
                shift = slot_spec["shift"]
                pad = slot_spec["pad"]
                previous_gap = gap_specs[slot_idx]
                next_gap = gap_specs[slot_idx + 1]
                previous_gap_row = previous_gap["row"]
                next_gap_row = next_gap["row"]
                leading_filler_clip = (
                    compact_clip(
                        previous_gap_row,
                        pad,
                        source_segments=source_segments_for_duration(
                            accepted_window_duration(previous_gap_row),
                            (
                                previous_gap["chain_start"]
                                + previous_gap["left_extension"]
                                + args.aligned_filler_gap_sec
                            ),
                            pad,
                        ),
                        connected_chain_offset_sec=(
                            previous_gap["left_extension"]
                            + args.aligned_filler_gap_sec
                        ),
                    )
                    if pad > 1e-6
                    else None
                )
                trailing_filler_clip = (
                    compact_clip(
                        next_gap_row,
                        pad,
                        source_segments=source_segments_for_duration(
                            accepted_window_duration(next_gap_row),
                            next_gap["chain_start"],
                            pad,
                        ),
                        connected_chain_offset_sec=0.0,
                    )
                    if pad > 1e-6
                    else None
                )

                if shift >= 0:
                    visual_gt = [[cursor, cursor + clip_len]]
                    audio_gt = [[cursor + pad, cursor + pad + clip_len]]
                    filler_modalities = ("audio", "visual")
                else:
                    audio_gt = [[cursor, cursor + clip_len]]
                    visual_gt = [[cursor + pad, cursor + pad + clip_len]]
                    filler_modalities = ("visual", "audio")

                filler_intervals = []
                if pad > 1e-6:
                    filler_intervals = [
                        {
                            "modality": filler_modalities[0],
                            "interval": [round(cursor, 3), round(cursor + pad, 3)],
                            "filler_label": previous_gap_row["label"],
                            "filler_clip": leading_filler_clip,
                            "connected_aligned_gap": "previous",
                        },
                        {
                            "modality": filler_modalities[1],
                            "interval": [
                                round(cursor + clip_len, 3),
                                round(cursor + clip_len + pad, 3),
                            ],
                            "filler_label": next_gap_row["label"],
                            "filler_clip": trailing_filler_clip,
                            "connected_aligned_gap": "next",
                        },
                    ]

                visual_gt = [
                    [round(start, 3), round(end, 3)]
                    for start, end in visual_gt
                ]
                audio_gt = [
                    [round(start, 3), round(end, 3)]
                    for start, end in audio_gt
                ]
                total_gt = interval_union(visual_gt + audio_gt)
                av_overlap_gt = interval_intersection(visual_gt, audio_gt)
                event_clip = compact_clip(clip, clip_len, crop_start)
                slots.append({
                    "slot": slot_idx,
                    "clip": event_clip,
                    "label": clip["label"],
                    "temporal_relation": relation,
                    "clip_duration_sec": round(clip_len, 3),
                    "audio_shift_sec": round(shift, 3),
                    "visual_gt": visual_gt,
                    "audio_gt": audio_gt,
                    "total_gt": total_gt,
                    "av_overlap_gt": av_overlap_gt,
                    "leading_filler_label": (
                        previous_gap_row["label"] if pad > 1e-6 else None
                    ),
                    "leading_filler_clip": leading_filler_clip,
                    "trailing_filler_label": (
                        next_gap_row["label"] if pad > 1e-6 else None
                    ),
                    "trailing_filler_clip": trailing_filler_clip,
                    "filler_intervals": filler_intervals,
                })
                part_plan.append({
                    "kind": "event_with_shift_filler",
                    "slot": slot_idx,
                    "event_clip": event_clip,
                    "prev_gap_clip": leading_filler_clip,
                    "next_gap_clip": trailing_filler_clip,
                    "duration": round(clip_len, 3),
                    "pad_duration": round(pad, 3),
                    "shift": round(shift, 3),
                    "temporal_relation": relation,
                })
                cursor += clip_len + pad
                if slot_idx != n_slots - 1:
                    add_aligned_gap(
                        slot_idx + 1,
                        "between_slots",
                        after_slot=slot_idx,
                    )
            add_aligned_gap(n_slots, "end", after_slot=n_slots - 1)

            if cursor > args.max_total_sec:
                continue
            by_label = trial_by_label
            if relation_quota is not None:
                relation_quota = trial_relation_quota

            sample_id = (
                f"{split_name}_s{sid_idx:06d}_{SAMPLE_CONDITION}"
                f"_n{len(slots):02d}_dur{int(round(cursor)):03d}"
            )
            relative_video = (
                Path("videos")
                / split_name
                / SAMPLE_CONDITION
                / f"{sample_id}.mp4"
            )
            sample = {
                "schema_version": SAMPLE_SCHEMA_VERSION,
                "dataset_version": DATASET_VERSION,
                "sample_id": sample_id,
                "split": split_name,
                "video": str(relative_video),
                "condition": SAMPLE_CONDITION,
                "num_slots": len(slots),
                "duration_sec": round(cursor, 3),
                "video_format": {
                    "width": args.width,
                    "height": args.height,
                    "fps": args.fps,
                    "aspect_ratio": "16:9",
                },
                "condition_sampling": "mixed_only_per_sample",
                "slot_relation_sampling": (
                    "split_level_quota_"
                    "aligned_audio_delay_audio_lead_ratio_1_1_1"
                ),
                "clip_duration_sampling": (
                    "continuous_uniform_5_10_sec_with_uniform_crop_"
                    "inside_accepted_window"
                ),
                "shift_sampling": (
                    "aligned_zero_else_uniform_0p5_to_"
                    "min_0p8_clip_duration_clip_minus_0p5"
                ),
                "explicit_random_gap": False,
                "aligned_filler_gap_sec": args.aligned_filler_gap_sec,
                "filler_policy": (
                    "connected_shift_filler_plus_fixed_av_aligned_gap_"
                    "with_single_wrap_when_chain_exceeds_accepted_window"
                ),
                "label_policy": (
                    "unique_exact_labels_and_reviewed_conflict_pairs_"
                    "blocked_within_sample"
                ),
                "aligned_filler_gaps": aligned_filler_gaps,
                "slots": slots,
                "part_plan": part_plan,
            }
            break

        if sample is None:
            raise RuntimeError(f"failed to build {split_name} sample {sid_idx}")
        samples.append(sample)
        for slot in sample["slots"]:
            queries.append({
                "schema_version": QUERY_SCHEMA_VERSION,
                "dataset_version": DATASET_VERSION,
                "query_id": f"{sample['sample_id']}_q{slot['slot']:02d}",
                "sample_id": sample["sample_id"],
                "split": split_name,
                "video": sample["video"],
                "query": slot["label"],
                "slot": slot["slot"],
                "condition": SAMPLE_CONDITION,
                "num_slots": sample["num_slots"],
                "duration_sec": sample["duration_sec"],
                "clip_duration_sec": slot["clip_duration_sec"],
                "temporal_relation": slot["temporal_relation"],
                "audio_shift_sec": slot["audio_shift_sec"],
                "visual_gt": slot["visual_gt"],
                "audio_gt": slot["audio_gt"],
                "total_gt": slot["total_gt"],
                "av_overlap_gt": slot["av_overlap_gt"],
                "shift_filler_labels": [
                    item["filler_label"]
                    for item in slot["filler_intervals"]
                ],
                "filler_intervals": slot["filler_intervals"],
                "clip_index": slot["clip"]["clip_index"],
                "youtube_id": slot["clip"]["youtube_id"],
            })
    return samples, queries, flatten_by_label(by_label)


def ffmpeg_number(value):
    return f"{float(value):.6f}"


def video_filter(parts, idx, label, source_start, duration, width, height, fps):
    source_start = ffmpeg_number(source_start)
    duration = ffmpeg_number(duration)
    parts.append(
        f"[{idx}:v]trim=start={source_start}:duration={duration},setpts=PTS-STARTPTS,"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p[{label}]"
    )


def audio_filter(parts, idx, label, source_start, duration):
    source_start = ffmpeg_number(source_start)
    duration = ffmpeg_number(duration)
    parts.append(
        f"[{idx}:a]atrim=start={source_start}:duration={duration},asetpts=PTS-STARTPTS,"
        f"pan=stereo|FL=c0|FR=c0,aresample=48000,aformat=sample_fmts=fltp,"
        f"apad=whole_dur={duration},atrim=start=0:duration={duration},asetpts=PTS-STARTPTS[{label}]"
    )


def clip_source_segments(clip):
    segments = clip.get("source_segments")
    if segments:
        segments = [dict(segment) for segment in segments]
    else:
        segments = [{
        "start_sec": float(clip.get("source_crop_start_sec", 0.0)),
        "duration_sec": float(clip["duration"]),
        "end_sec": (
            float(clip.get("source_crop_start_sec", 0.0))
            + float(clip["duration"])
        ),
        }]

    # A wrapped segment shorter than one 25-fps frame yields no video frame.
    # Rebalance only such boundary fragments while preserving total duration.
    min_render_sec = 1.0 / 25.0
    if len(segments) > 1:
        for idx, segment in enumerate(segments):
            duration = float(segment["duration_sec"])
            if duration >= min_render_sec:
                continue
            deficit = min_render_sec - duration
            if idx > 0 and float(segments[idx - 1]["duration_sec"]) > deficit:
                previous = segments[idx - 1]
                previous["duration_sec"] = float(previous["duration_sec"]) - deficit
                previous["end_sec"] = (
                    float(previous["start_sec"]) + float(previous["duration_sec"])
                )
                segment["duration_sec"] = min_render_sec
                segment["end_sec"] = (
                    float(segment["start_sec"]) + min_render_sec
                )
            elif idx + 1 < len(segments) and float(segments[idx + 1]["duration_sec"]) > deficit:
                following = segments[idx + 1]
                segment["start_sec"] = max(
                    0.0, float(segment["start_sec"]) - deficit
                )
                segment["duration_sec"] = min_render_sec
                segment["end_sec"] = (
                    float(segment["start_sec"]) + min_render_sec
                )
                following["duration_sec"] = (
                    float(following["duration_sec"]) - deficit
                )
                following["end_sec"] = (
                    float(following["start_sec"])
                    + float(following["duration_sec"])
                )
    return segments


def video_clip_filter(parts, idx, label, clip, width, height, fps):
    segments = clip_source_segments(clip)
    if len(segments) == 1:
        segment = segments[0]
        video_filter(
            parts,
            idx,
            label,
            float(segment["start_sec"]),
            float(segment["duration_sec"]),
            width,
            height,
            fps,
        )
        return
    segment_labels = []
    for segment_idx, segment in enumerate(segments):
        segment_label = f"{label}s{segment_idx}"
        video_filter(
            parts,
            idx,
            segment_label,
            float(segment["start_sec"]),
            float(segment["duration_sec"]),
            width,
            height,
            fps,
        )
        segment_labels.append(f"[{segment_label}]")
    parts.append(
        "".join(segment_labels)
        + f"concat=n={len(segment_labels)}:v=1:a=0[{label}]"
    )


def audio_clip_filter(parts, idx, label, clip):
    segments = clip_source_segments(clip)
    if len(segments) == 1:
        segment = segments[0]
        audio_filter(
            parts,
            idx,
            label,
            float(segment["start_sec"]),
            float(segment["duration_sec"]),
        )
        return
    segment_labels = []
    for segment_idx, segment in enumerate(segments):
        segment_label = f"{label}s{segment_idx}"
        audio_filter(
            parts,
            idx,
            segment_label,
            float(segment["start_sec"]),
            float(segment["duration_sec"]),
        )
        segment_labels.append(f"[{segment_label}]")
    parts.append(
        "".join(segment_labels)
        + f"concat=n={len(segment_labels)}:v=0:a=1[{label}]"
    )


def synthesize_one(sample, out_root, args):
    out_path = out_root / sample["video"]
    meta_path = out_path.with_suffix(".json")
    if out_path.exists() and meta_path.exists() and not args.overwrite:
        return {"sample_id": sample["sample_id"], "status": "exists"}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = []
    input_map = {}

    def add_input(path):
        path = str(path)
        if path not in input_map:
            input_map[path] = len(input_map)
            inputs.extend(["-i", path])
        return input_map[path]

    filters, vparts, aparts = [], [], []
    part_idx = 0
    for part in sample["part_plan"]:
        dur = float(part["duration"])
        if part["kind"] == "aligned_filler_gap":
            idx = add_input(part["clip"]["clip_path"])
            gap_v, gap_a = f"v{part_idx}g", f"a{part_idx}g"
            video_clip_filter(
                filters,
                idx,
                gap_v,
                part["clip"],
                args.width,
                args.height,
                args.fps,
            )
            audio_clip_filter(filters, idx, gap_a, part["clip"])
            vparts.append(f"[{gap_v}]")
            aparts.append(f"[{gap_a}]")
            part_idx += 1
            continue
        pad = float(part["pad_duration"])
        shift = float(part["shift"])
        eidx = add_input(part["event_clip"]["clip_path"])
        ev_v, ev_a = f"v{part_idx}", f"a{part_idx}"
        video_clip_filter(
            filters,
            eidx,
            ev_v,
            part["event_clip"],
            args.width,
            args.height,
            args.fps,
        )
        audio_clip_filter(filters, eidx, ev_a, part["event_clip"])
        if pad > 1e-6:
            if shift > 0:
                next_idx = add_input(part["next_gap_clip"]["clip_path"])
                prev_idx = add_input(part["prev_gap_clip"]["clip_path"])
                next_v = f"v{part_idx}next"
                prev_a = f"a{part_idx}prev"
                video_clip_filter(
                    filters,
                    next_idx,
                    next_v,
                    part["next_gap_clip"],
                    args.width,
                    args.height,
                    args.fps,
                )
                audio_clip_filter(
                    filters,
                    prev_idx,
                    prev_a,
                    part["prev_gap_clip"],
                )
                vparts.extend([f"[{ev_v}]", f"[{next_v}]"])
                aparts.extend([f"[{prev_a}]", f"[{ev_a}]"])
            else:
                prev_idx = add_input(part["prev_gap_clip"]["clip_path"])
                next_idx = add_input(part["next_gap_clip"]["clip_path"])
                prev_v = f"v{part_idx}prev"
                next_a = f"a{part_idx}next"
                video_clip_filter(
                    filters,
                    prev_idx,
                    prev_v,
                    part["prev_gap_clip"],
                    args.width,
                    args.height,
                    args.fps,
                )
                audio_clip_filter(
                    filters,
                    next_idx,
                    next_a,
                    part["next_gap_clip"],
                )
                vparts.extend([f"[{prev_v}]", f"[{ev_v}]"])
                aparts.extend([f"[{ev_a}]", f"[{next_a}]"])
        else:
            vparts.append(f"[{ev_v}]")
            aparts.append(f"[{ev_a}]")
        part_idx += 1

    filters.append("".join(vparts) + f"concat=n={len(vparts)}:v=1:a=0[vout]")
    filters.append("".join(aparts) + f"concat=n={len(aparts)}:v=0:a=1[aout]")
    tmp = out_path.with_suffix(".tmp.mp4")
    cmd = [
        str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
        *inputs, "-filter_complex", ";".join(filters), "-map", "[vout]", "-map", "[aout]",
        "-t", f"{float(sample['duration_sec']):.3f}",
        "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", args.audio_bitrate,
        "-movflags", "+faststart", str(tmp),
    ]
    run(cmd)
    tmp.replace(out_path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return {"sample_id": sample["sample_id"], "status": "ok"}


def target_clip_indices(samples):
    used = []
    for sample in samples:
        for slot in sample["slots"]:
            used.append(("target", sample["sample_id"], slot["clip"]["clip_index"]))
    return used


def validate(samples, block_pairs):
    errors = []

    def validate_crop(sample_id, context, clip):
        if not clip:
            return
        accepted_duration = float(clip.get("accepted_window_duration_sec", 0.0))
        segments = clip_source_segments(clip)
        total_duration = 0.0
        for segment_idx, segment in enumerate(segments):
            start = float(segment["start_sec"])
            duration = float(segment["duration_sec"])
            end = float(segment["end_sec"])
            total_duration += duration
            if (
                start < -1e-6
                or duration <= 0
                or abs(end - (start + duration)) > 1e-4
                or end > accepted_duration + 1e-6
            ):
                errors.append([
                    sample_id,
                    "source_segment_outside_accepted_window",
                    context,
                    segment_idx,
                    segment,
                    accepted_duration,
                ])
        if abs(total_duration - float(clip["duration"])) > 0.002:
            errors.append([
                sample_id,
                "source_segment_duration_mismatch",
                context,
                total_duration,
                clip["duration"],
            ])
        if int(clip.get("source_wrap_count", 0)) != max(0, len(segments) - 1):
            errors.append([
                sample_id,
                "source_wrap_count_mismatch",
                context,
            ])

    for sample in samples:
        structural_labels = []
        structural_clip_indices = []
        gaps = sample.get("aligned_filler_gaps", [])
        for gap_idx, gap in enumerate(gaps):
            structural_labels.append(gap["filler_label"])
            structural_clip_indices.append(gap["filler_clip"]["clip_index"])
            validate_crop(sample["sample_id"], f"gap_{gap_idx}", gap["filler_clip"])
            chain = gap.get("connected_source_chain", {})
            left = float(chain.get("left_shift_extension_sec", 0.0))
            aligned = float(chain.get("aligned_gap_sec", gap["duration"]))
            right = float(chain.get("right_shift_extension_sec", 0.0))
            chain_duration = sum(
                float(segment["duration_sec"])
                for segment in chain.get("source_segments", [])
            )
            if abs(chain_duration - (left + aligned + right)) > 0.002:
                errors.append([
                    sample["sample_id"],
                    "bad_connected_filler_chain_length",
                    gap_idx,
                    chain,
                ])
            if gap_idx > 0 and left > 1e-6:
                previous_trailing = sample["slots"][gap_idx - 1]["trailing_filler_clip"]
                if (
                    previous_trailing is None
                    or abs(float(previous_trailing["connected_chain_offset_sec"])) > 0.002
                    or abs(float(previous_trailing["duration"]) - left) > 0.002
                ):
                    errors.append([
                        sample["sample_id"],
                        "discontinuous_left_gap_extension",
                        gap_idx,
                    ])
            if gap_idx < len(sample["slots"]) and right > 1e-6:
                following_leading = sample["slots"][gap_idx]["leading_filler_clip"]
                if (
                    following_leading is None
                    or abs(
                        float(following_leading["connected_chain_offset_sec"])
                        - (left + aligned)
                    ) > 0.002
                    or abs(float(following_leading["duration"]) - right) > 0.002
                ):
                    errors.append([
                        sample["sample_id"],
                        "discontinuous_right_gap_extension",
                        gap_idx,
                    ])
        for slot in sample["slots"]:
            structural_labels.append(slot["label"])
            structural_clip_indices.append(slot["clip"]["clip_index"])
            validate_crop(
                sample["sample_id"],
                f"slot_{slot['slot']}_target",
                slot["clip"],
            )
            validate_crop(
                sample["sample_id"],
                f"slot_{slot['slot']}_leading_filler",
                slot.get("leading_filler_clip"),
            )
            validate_crop(
                sample["sample_id"],
                f"slot_{slot['slot']}_trailing_filler",
                slot.get("trailing_filler_clip"),
            )
            if interval_union(slot["visual_gt"] + slot["audio_gt"]) != slot["total_gt"]:
                errors.append([sample["sample_id"], "bad_total_gt", slot["slot"]])
            relation = slot.get("temporal_relation")
            shift = float(slot["audio_shift_sec"])
            if relation == "aligned" and abs(shift) > 1e-6:
                errors.append([sample["sample_id"], "aligned_nonzero_shift", slot["slot"], shift])
            if relation == "audio_delay" and shift < 0.5:
                errors.append([sample["sample_id"], "delay_shift_invalid", slot["slot"], shift])
            if relation == "audio_lead" and shift > -0.5:
                errors.append([sample["sample_id"], "lead_shift_invalid", slot["slot"], shift])
            for item in slot["filler_intervals"]:
                a, b = item["interval"]
                if b <= a:
                    errors.append([sample["sample_id"], "bad_filler_interval", slot["slot"], item])
                if item.get("connected_aligned_gap") == "previous":
                    expected = sample["aligned_filler_gaps"][slot["slot"]]["filler_label"]
                elif item.get("connected_aligned_gap") == "next":
                    expected = sample["aligned_filler_gaps"][slot["slot"] + 1]["filler_label"]
                else:
                    expected = None
                if expected and item.get("filler_label") != expected:
                    errors.append([sample["sample_id"], "disconnected_shift_filler", slot["slot"], item.get("filler_label"), expected])
        if len(structural_labels) != len(set(structural_labels)):
            errors.append([sample["sample_id"], "duplicate_structural_label", structural_labels])
        if len(structural_clip_indices) != len(set(structural_clip_indices)):
            errors.append([
                sample["sample_id"],
                "duplicate_target_or_gap_clip_index",
                structural_clip_indices,
            ])
        for gap in gaps:
            if gap["filler_clip"].get("source_split") != sample["split"]:
                errors.append([
                    sample["sample_id"],
                    "cross_split_filler",
                    gap["filler_clip"].get("source_split"),
                    sample["split"],
                ])
        for i, a in enumerate(structural_labels):
            for b in structural_labels[i + 1:]:
                if tuple(sorted((a, b))) in block_pairs:
                    errors.append([sample["sample_id"], "blocked_pair", a, b])
        if float(sample["duration_sec"]) > 120.0001:
            errors.append([sample["sample_id"], "duration_gt_120", sample["duration_sec"]])
    used = target_clip_indices(samples)
    counts = Counter(idx for _, _, idx in used)
    dupes = [idx for idx, count in counts.items() if count > 1]
    if dupes:
        errors.append(["global_duplicate_target_clip_index", dupes[:20], len(dupes)])
    return errors


def validate_queries(samples, queries):
    errors = []
    expected = {}
    filler_labels_by_sample = {}
    for sample in samples:
        filler_labels_by_sample[sample["sample_id"]] = {
            gap["filler_label"]
            for gap in sample.get("aligned_filler_gaps", [])
        }
        for slot in sample["slots"]:
            expected[(sample["sample_id"], slot["slot"])] = slot["label"]
    actual = {}
    for query in queries:
        key = (query["sample_id"], query["slot"])
        if key in actual:
            errors.append(["duplicate_query_for_target_slot", key])
        actual[key] = query["query"]
        if query["query"] in filler_labels_by_sample.get(query["sample_id"], set()):
            errors.append([
                query["sample_id"],
                "filler_label_used_as_query",
                query["query"],
            ])
    if set(expected) != set(actual):
        errors.append([
            "query_target_key_mismatch",
            len(expected),
            len(actual),
            len(set(expected) - set(actual)),
            len(set(actual) - set(expected)),
        ])
    for key, label in expected.items():
        if actual.get(key) != label:
            errors.append([
                "query_target_label_mismatch",
                key,
                label,
                actual.get(key),
            ])
    return errors


def make_swift_rows(queries, out_root):
    rows = []
    for q in queries:
        video_abs = str((out_root / q["video"]).resolve())
        solution = {
            "intervals": q["total_gt"],
            "visual_gt": q["visual_gt"],
            "audio_gt": q["audio_gt"],
            "av_overlap_gt": q["av_overlap_gt"],
            "duration_sec": q["duration_sec"],
            "query_id": q["query_id"],
        }
        rows.append({
            "schema_version": "avrtg.grpo.v1",
            "dataset_version": DATASET_VERSION,
            "prompt_policy_id": PROMPT_POLICY_ID,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(query=q["query"])},
            ],
            "videos": [video_abs],
            "solution": json.dumps(solution, ensure_ascii=False),
            "query_id": q["query_id"],
            "sample_id": q["sample_id"],
            "query": q["query"],
            "condition": q["condition"],
            "temporal_relation": q["temporal_relation"],
            "duration_sec": q["duration_sec"],
        })
    return rows


def summarize(out_root, rows, split_counts, samples, queries, errors, args, requested_samples, unused_by_split):
    condition_counts = defaultdict(lambda: defaultdict(int))
    slot_relation_counts = defaultdict(lambda: defaultdict(int))
    durations, shifts = [], []
    for s in samples:
        condition_counts[s["split"]][s["condition"]] += 1
        durations.append(float(s["duration_sec"]))
        for slot in s["slots"]:
            slot_relation_counts[s["split"]][slot["temporal_relation"]] += 1
            if abs(float(slot["audio_shift_sec"])) > 1e-6:
                shifts.append(abs(float(slot["audio_shift_sec"])))
    used_records = target_clip_indices(samples)
    used_total = len({idx for _, _, idx in used_records})
    unused_counts = {split: len(items) for split, items in unused_by_split.items()}
    filler_gap_events = sum(len(s.get("aligned_filler_gaps", [])) for s in samples)
    filler_unique_rows = len({
        gap["filler_clip"]["clip_index"]
        for s in samples
        for gap in s.get("aligned_filler_gaps", [])
    })
    wrapped_filler_chains = sum(
        int(gap.get("connected_source_chain", {}).get("wrap_count", 0)) > 0
        for sample in samples
        for gap in sample.get("aligned_filler_gaps", [])
    )
    summary = {
        "dataset": out_root.name,
        "version": DATASET_VERSION,
        "sample_schema_version": SAMPLE_SCHEMA_VERSION,
        "query_schema_version": QUERY_SCHEMA_VERSION,
        "prompt_policy_id": PROMPT_POLICY_ID,
        "source_pool": {
            "accepted_csv": str(args.accepted_csv),
            "accepted_csv_sha256": args.accepted_csv_sha256,
            "expected_rows": args.expected_source_rows,
            "loaded_rows": len(rows),
            "block_csv": str(args.block_csv),
            "block_csv_sha256": args.block_csv_sha256,
        },
        "random_seed": args.seed,
        "requested_samples": requested_samples,
        "source_rows": len(rows),
        "clip_usage_policy": "all_source_clip_rows_used_once_as_target_query; fillers_are_split_local_reusable_context" if args.exhaustive else "fixed_requested_sample_count",
        "target_unique_clip_rows": used_total,
        "unused_target_clip_rows": sum(unused_counts.values()),
        "unused_target_clip_rows_by_split": unused_counts,
        "target_clip_rows": len(queries),
        "aligned_filler_gap_events": filler_gap_events,
        "aligned_filler_gap_unique_clip_rows_reused": filler_unique_rows,
        "connected_filler_chains_with_single_wrap": wrapped_filler_chains,
        "label_split_policy": "lt3=train, 3-4=train+val, ge5=approximately 8:1:1 with minimum val/test coverage",
        "temporal_overlap_split_policy": "overlapping source intervals are grouped in train first; same-label non-overlapping train clips are swapped to preserve exact split counts",
        "temporal_overlap_components_repaired": getattr(args, "temporal_overlap_components_repaired", 0),
        "cross_split_temporal_overlap_pairs": getattr(args, "cross_split_temporal_overlap_pairs", None),
        "condition_ratio_target": "mixed_temporal_shift only",
        "slot_relation_ratio_target": "aligned:audio_delay:audio_lead = 1:1:1 per split",
        "samples": len(samples),
        "queries": len(queries),
        "condition_counts": {k: dict(v) for k, v in condition_counts.items()},
        "slot_relation_counts": {k: dict(v) for k, v in slot_relation_counts.items()},
        "duration_sec": {
            "min": round(min(durations), 3) if durations else None,
            "max": round(max(durations), 3) if durations else None,
            "mean": round(sum(durations) / len(durations), 3) if durations else None,
        },
        "abs_nonzero_shift_sec": {
            "min": round(min(shifts), 3) if shifts else None,
            "max": round(max(shifts), 3) if shifts else None,
            "mean": round(sum(shifts) / len(shifts), 3) if shifts else None,
        },
        "video_format": {"width": args.width, "height": args.height, "fps": args.fps, "aspect_ratio": "16:9"},
        "explicit_random_gap": False,
        "aligned_filler_gap_sec": args.aligned_filler_gap_sec,
        "target_crop_policy": "uniform_crop_inside_human_validated_accepted_window",
        "filler_policy": "connected_shift_filler_plus_fixed_av_aligned_gap_with_single_wrap_when_chain_exceeds_accepted_window",
        "validation_errors": len(errors),
        "validation_error_examples": errors[:20],
        "label_split_count_examples": dict(list(split_counts.items())[:10]),
    }
    metadata_root = out_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    with (metadata_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted-csv", type=Path, required=True)
    ap.add_argument("--clips-dir", type=Path, help="Override clip_path directories, retaining each filename")
    ap.add_argument("--expected-source-rows", type=int, default=EXPECTED_ACCEPTED_ROWS)
    ap.add_argument("--expected-source-sha256", default=EXPECTED_ACCEPTED_SHA256)
    ap.add_argument("--block-csv", type=Path, default=DEFAULT_BLOCKS)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260719)
    ap.add_argument("--review-samples-per-split", type=int, default=3)
    ap.add_argument("--samples-train", type=int)
    ap.add_argument("--samples-val", type=int)
    ap.add_argument("--samples-test", type=int)
    ap.add_argument("--exhaustive", action="store_true", help="Use every accepted clip exactly once as a target/query clip.")
    ap.add_argument("--min-slots", type=int, default=5)
    ap.add_argument("--max-slots", type=int, default=9)
    ap.add_argument("--min-clip-sec", type=float, default=5.0)
    ap.add_argument("--max-clip-sec", type=float, default=10.0)
    ap.add_argument("--min-shift-sec", type=float, default=0.5)
    ap.add_argument("--max-shift-frac", type=float, default=0.8)
    ap.add_argument("--min-overlap-sec", type=float, default=0.5)
    ap.add_argument("--aligned-filler-gap-sec", type=float, default=3.0)
    ap.add_argument("--max-total-sec", type=float, default=120.0)
    ap.add_argument("--max-attempts", type=int, default=500)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--audio-bitrate", default="128k")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()

    args.accepted_csv = validate_source_path(args.accepted_csv)
    args.block_csv = Path(args.block_csv).expanduser().resolve()
    args.accepted_csv_sha256 = sha256_file(args.accepted_csv)
    args.block_csv_sha256 = sha256_file(args.block_csv)
    if args.expected_source_sha256 and args.accepted_csv_sha256 != args.expected_source_sha256:
        raise RuntimeError(
            "accepted source SHA-256 mismatch: "
            f"expected {args.expected_source_sha256}, got {args.accepted_csv_sha256}"
        )

    out_root = args.out_root
    if out_root.exists() and any(out_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out_root}; choose a new directory")
    out_root.mkdir(parents=True, exist_ok=True)
    metadata_root = out_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    schema_contract = {
        "dataset_version": DATASET_VERSION,
        "canonical_manifests": {
            "sample": {
                "schema_version": SAMPLE_SCHEMA_VERSION,
                "files": ["samples_manifest_train.jsonl", "samples_manifest_val.jsonl", "samples_manifest_test.jsonl"],
            },
            "query": {
                "schema_version": QUERY_SCHEMA_VERSION,
                "files": ["queries_manifest_train.jsonl", "queries_manifest_val.jsonl", "queries_manifest_test.jsonl"],
                "model_target": "total_gt = union(audio_gt, visual_gt)",
                "diagnostic_fields": ["audio_gt", "visual_gt", "av_overlap_gt"],
            },
        },
        "prompt": {
            "policy_id": PROMPT_POLICY_ID,
            "system": SYSTEM_PROMPT,
            "user_template": USER_PROMPT_TEMPLATE,
            "thinking_trace": False,
        },
        "model_output": {
            "content_type": "strict_json_inside_answer_tags",
            "template": '<answer>{"intervals":[[start_second,end_second], ...]}</answer>',
            "top_level_keys": ["intervals"],
        },
        "derived_manifests": {
            "grpo": {
                "files": ["swift_rlvr_train.jsonl", "swift_rlvr_val.jsonl", "swift_rlvr_test.jsonl"],
                "solution_intervals_source": "query.total_gt",
                "diagnostic_metadata_not_exposed_in_prompt": True,
            }
        },
    }
    (metadata_root / "schema_contract.json").write_text(
        json.dumps(schema_contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = read_rows(args.accepted_csv, args.clips_dir)
    if len(rows) != args.expected_source_rows:
        raise RuntimeError(
            "accepted source row/path validation mismatch: "
            f"expected {args.expected_source_rows}, loaded {len(rows)}"
        )
    clip_indices = [row["clip_index"] for row in rows]
    if len(set(clip_indices)) != len(clip_indices):
        raise RuntimeError("accepted source contains duplicate clip_index values")
    block_pairs = read_block_pairs(args.block_csv)
    splits, split_counts, overlap_repair_report = split_rows_by_label(rows, args.seed)
    cross_split_overlaps = find_cross_split_temporal_overlaps(splits)
    args.temporal_overlap_components_repaired = sum(
        len(item["components"]) for item in overlap_repair_report
    )
    args.cross_split_temporal_overlap_pairs = len(cross_split_overlaps)
    write_jsonl(
        metadata_root / "temporal_overlap_split_repairs.jsonl",
        overlap_repair_report,
    )
    write_jsonl(
        metadata_root / "cross_split_temporal_overlaps.jsonl",
        cross_split_overlaps,
    )
    if cross_split_overlaps:
        raise RuntimeError(
            "cross-split temporal overlap remains after repair: "
            f"{cross_split_overlaps[:5]}"
        )

    all_samples, all_queries = [], []
    if args.exhaustive:
        requested_samples = {"train": "exhaustive", "val": "exhaustive", "test": "exhaustive"}
    else:
        requested_samples = {
            "train": args.samples_train if args.samples_train is not None else args.review_samples_per_split,
            "val": args.samples_val if args.samples_val is not None else args.review_samples_per_split,
            "test": args.samples_test if args.samples_test is not None else args.review_samples_per_split,
        }
    unused_by_split = {}
    for split_name in ["train", "val", "test"]:
        samples_per_split = None if args.exhaustive else requested_samples[split_name]
        samples, queries, unused = build_split(split_name, splits[split_name], block_pairs, args, samples_per_split)
        unused_by_split[split_name] = unused
        all_samples.extend(samples)
        all_queries.extend(queries)
        write_jsonl(metadata_root / f"samples_manifest_{split_name}.jsonl", samples)
        write_jsonl(metadata_root / f"queries_manifest_{split_name}.jsonl", queries)
        write_jsonl(metadata_root / f"unused_target_clip_rows_{split_name}.jsonl", unused)
    write_jsonl(metadata_root / "samples_manifest.jsonl", all_samples)
    write_jsonl(metadata_root / "queries_manifest.jsonl", all_queries)
    write_jsonl(metadata_root / "unused_target_clip_rows.jsonl", [row for rows_for_split in unused_by_split.values() for row in rows_for_split])

    errors = validate(all_samples, block_pairs)
    errors.extend(validate_queries(all_samples, all_queries))
    if args.exhaustive:
        unused_total = sum(len(items) for items in unused_by_split.values())
        if unused_total:
            errors.append(["unused_target_clip_rows", unused_total])
        if len(all_queries) != len(rows):
            errors.append(["target_query_count_mismatch", len(all_queries), len(rows)])
    summary = summarize(out_root, rows, split_counts, all_samples, all_queries, errors, args, requested_samples, unused_by_split)
    if errors:
        raise SystemExit(f"validation failed: {errors[:5]}")

    if not args.manifest_only:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(synthesize_one, sample, out_root, args) for sample in all_samples]
            for fut in as_completed(futs):
                fut.result()

    for split_name in ["train", "val", "test"]:
        split_queries = [q for q in all_queries if q["split"] == split_name]
        write_jsonl(metadata_root / f"swift_rlvr_{split_name}.jsonl", make_swift_rows(split_queries, out_root))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
