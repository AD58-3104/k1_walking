#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the latest TensorBoard reward scalars as a Markdown table."
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("logs/rsl_rl"),
        help="Root directory that contains experiment run directories.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Specific run directory to inspect. If omitted, the latest run under --log-root is used.",
    )
    parser.add_argument(
        "--event-file",
        type=Path,
        default=None,
        help="Specific TensorBoard event file to inspect.",
    )
    parser.add_argument(
        "--tag-prefix",
        nargs="+",
        default=["Train/", "Episode_Reward/"],
        help="Only include scalar tags starting with these prefixes. Can specify multiple.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="If set, show the latest scalar value recorded at or before this global step.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("name", "value", "step"),
        default="name",
        help="Column used to sort the output rows.",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort in descending order.",
    )
    return parser.parse_args()


def find_latest_event_file(log_root: Path, run_dir: Path | None) -> Path:
    search_root = run_dir if run_dir is not None else log_root
    if not search_root.exists():
        raise FileNotFoundError(f"Log directory not found: {search_root}")

    event_files = sorted(search_root.rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under: {search_root}")
    return event_files[-1]


def load_scalar_rows(event_file: Path, tag_prefixes: list[str], step: int | None = None) -> list[dict[str, float | int | str]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tensorboard is not installed. Install it with `pip install tensorboard` to read event files."
        ) from exc

    accumulator = event_accumulator.EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()

    tags = accumulator.Tags().get("scalars", [])

    def find_matching_prefix(tag: str) -> str | None:
        for prefix in tag_prefixes:
            if tag.startswith(prefix):
                return prefix
        return None

    rows: list[dict[str, float | int | str]] = []
    for tag in tags:
        matched_prefix = find_matching_prefix(tag)
        if matched_prefix is None:
            continue
        events = accumulator.Scalars(tag)
        if not events:
            continue
        if step is None:
            selected_event = events[-1]
        else:
            selected_event = None
            for event in events:
                if event.step > step:
                    break
                selected_event = event
            if selected_event is None:
                continue
        rows.append(
            {
                "name": tag[len(matched_prefix):] if matched_prefix else tag,
                "full_tag": tag,
                "step": int(selected_event.step),
                "value": float(selected_event.value),
            }
        )
    return rows


def sort_rows(rows: list[dict[str, float | int | str]], sort_by: str, descending: bool) -> list[dict[str, float | int | str]]:
    if sort_by == "name":
        key_fn = lambda row: str(row["name"])
    elif sort_by == "value":
        key_fn = lambda row: float(row["value"])
    else:
        key_fn = lambda row: int(row["step"])
    return sorted(rows, key=key_fn, reverse=descending)


def to_markdown_table(rows: list[dict[str, float | int | str]]) -> str:
    if not rows:
        return "No matching scalar tags were found."

    headers = ("Metric", "Step", "Value")
    table_rows = [
        (
            str(row["name"]),
            str(row["step"]),
            f"{row['value']:.6f}",
        )
        for row in rows
    ]
    widths = [
        max(len(headers[idx]), *(len(table_row[idx]) for table_row in table_rows))
        for idx in range(len(headers))
    ]

    lines = [
        f"| {headers[0]:<{widths[0]}} | {headers[1]:>{widths[1]}} | {headers[2]:>{widths[2]}} |",
        f"| {'-' * widths[0]} | {'-' * widths[1]}: | {'-' * widths[2]}: |",
    ]
    for name, step, value in table_rows:
        lines.append(f"| {name:<{widths[0]}} | {step:>{widths[1]}} | {value:>{widths[2]}} |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    try:
        event_file = args.event_file if args.event_file is not None else find_latest_event_file(args.log_root, args.run_dir)
        rows = load_scalar_rows(event_file, args.tag_prefix, args.step)
        rows = sort_rows(rows, args.sort_by, args.descending)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Event file: {event_file}")
    if args.step is not None:
        print(f"Requested step: {args.step} (showing the latest value at or before this step)")
    print(to_markdown_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
