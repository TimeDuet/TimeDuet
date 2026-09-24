import json
import math
import re
from typing import Any, Iterable

from swift.rewards import ORM, orms


Interval = list[float]
INVALID_REWARD = -0.50
GAP_LAMBDA = 0.30


def _load_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return {}
    return value


def _extract_payload(text: str) -> tuple[Any, bool]:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text or "", flags=re.I | re.S)
    if not match:
        return {}, False
    payload = match.group(1)
    try:
        return json.loads(payload), True
    except Exception:
        obj = re.search(r"\{.*\}", payload, flags=re.S)
        if obj:
            try:
                return json.loads(obj.group(0)), True
            except Exception:
                pass
    return {}, False


def _normalize(value: Any, duration: float | None) -> list[Interval]:
    value = _load_json(value)
    if isinstance(value, dict):
        value = value.get("intervals", [])
    if not isinstance(value, list):
        return []
    cleaned: list[Interval] = []
    for item in value:
        if isinstance(item, dict):
            start, end = item.get("start", item.get("start_second")), item.get("end", item.get("end_second"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[:2]
        else:
            continue
        try:
            start, end = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        if end < start:
            start, end = end, start
        if duration is not None:
            start, end = max(0.0, min(duration, start)), max(0.0, min(duration, end))
        if end - start >= 0.05:
            cleaned.append([start, end])
    cleaned.sort()
    merged: list[Interval] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _length(intervals: Iterable[Interval]) -> float:
    return sum(max(0.0, end - start) for start, end in intervals)


def _intersection(a: list[Interval], b: list[Interval]) -> float:
    total = 0.0
    i = j = 0
    while i < len(a) and j < len(b):
        start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if end > start:
            total += end - start
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _iou(prediction: list[Interval], target: list[Interval]) -> float:
    union = _length(prediction) + _length(target) - _intersection(prediction, target)
    return _intersection(prediction, target) / union if union > 0 else 0.0


def _difference(source: list[Interval], subtract: list[Interval]) -> list[Interval]:
    result: list[Interval] = []
    for start, end in source:
        fragments = [[start, end]]
        for cut_start, cut_end in subtract:
            next_fragments: list[Interval] = []
            for left, right in fragments:
                if cut_end <= left or cut_start >= right:
                    next_fragments.append([left, right])
                else:
                    if left < cut_start:
                        next_fragments.append([left, min(cut_start, right)])
                    if cut_end < right:
                        next_fragments.append([max(cut_end, left), right])
            fragments = next_fragments
        result.extend(fragments)
    return _normalize(result, None)


class TimeDuetReward(ORM):
    """r = .5*T-IoU + .5*min(A-IoU, V-IoU) - .30*abs(A-IoU - V-IoU)."""

    def __call__(self, completions, solution, **kwargs) -> list[float]:
        rewards: list[float] = []
        for completion, raw_solution in zip(completions, solution):
            try:
                target = _load_json(raw_solution)
                duration = float(target["duration_sec"])
                total = _normalize(target.get("intervals", []), duration)
                audio = _normalize(target.get("audio_gt", []), duration)
                visual = _normalize(target.get("visual_gt", []), duration)
                payload, parsed = _extract_payload(completion or "")
                prediction = _normalize(payload, duration)
                if not parsed or not prediction or not total or not audio or not visual:
                    rewards.append(INVALID_REWARD)
                    continue
                total_iou = _iou(prediction, total)
                audio_iou = _iou(_difference(prediction, _difference(visual, audio)), audio)
                visual_iou = _iou(_difference(prediction, _difference(audio, visual)), visual)
                reward = 0.5 * total_iou + 0.5 * min(audio_iou, visual_iou) - GAP_LAMBDA * abs(audio_iou - visual_iou)
                rewards.append(reward)
            except Exception:
                rewards.append(INVALID_REWARD)
        return rewards


orms["timeduet"] = TimeDuetReward
