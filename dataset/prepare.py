"""Prepare SFT and GRPO manifests from a downloaded TimeDuet dataset."""

import argparse
import json
from pathlib import Path

from dataset.build import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, interval_union


def read_split(root, split):
    path = root / "data" / f"{split}.jsonl"
    if not path.is_file():
        name = "val" if split == "validation" else split
        path = root / "metadata" / f"queries_manifest_{name}.jsonl"
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ids = [row["query_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError(f"Empty split or duplicate query IDs: {path}")
    return rows


def prepare_row(row, root, assistant=False):
    video = root / row["video"]
    if not video.is_file():
        raise FileNotFoundError(video)
    total = interval_union(row["audio_gt"] + row["visual_gt"])
    if total != row["total_gt"]:
        raise ValueError(f"Ground-truth union mismatch: {row['query_id']}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(query=row["query"])},
    ]
    solution = {key: row[key] for key in
                ("audio_gt", "visual_gt", "av_overlap_gt", "duration_sec", "query_id")}
    solution["intervals"] = total
    if assistant:
        answer = json.dumps({"intervals": total}, separators=(",", ":"))
        messages.append({"role": "assistant", "content": f"<answer>{answer}</answer>"})
    return {
        **row, "messages": messages, "videos": [str(video.resolve())],
        "solution": json.dumps(solution, ensure_ascii=False),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_split(args.data, "train")
    for name in ("sft.jsonl", "grpo.jsonl"):
        if (args.output / name).exists():
            raise FileExistsError(args.output / name)
    for row in rows:
        prepare_row(row, args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, selected, assistant in (
        ("sft.jsonl", rows, True),
        ("grpo.jsonl", [r for r in rows if abs(float(r["audio_shift_sec"])) > 1e-9], False),
    ):
        path = args.output / name
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new output directory")
        with path.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(prepare_row(row, args.data, assistant), ensure_ascii=False) + "\n")
        print(f"{path}: {len(selected)} queries")


if __name__ == "__main__":
    main()
