#!/usr/bin/env python3
"""Measure training-sample token lengths and optionally delete long samples."""

from __future__ import annotations

import argparse
import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Stats:
    valid: int = 0
    over_report: int = 0
    over_delete: int = 0
    over_very_long: int = 0
    deleted: int = 0
    errors: int = 0
    maximum: int = 0

    def merge(self, other: "Stats") -> None:
        for field in self.__dataclass_fields__:
            if field == "maximum":
                continue
            setattr(self, field, getattr(self, field) + getattr(other, field))
        self.maximum = max(self.maximum, other.maximum)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count assistant-response tokens in per-sample JSON files. By default "
            "this is a read-only report; pass --delete to remove overlong files."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Tokenizer model ID or local initialized model directory.",
    )
    parser.add_argument(
        "--dataset-dir",
        action="append",
        required=True,
        type=Path,
        help="Directory containing split JSON samples; repeat for multiple datasets.",
    )
    parser.add_argument("--report-limit", type=int, default=6144)
    parser.add_argument("--delete-limit", type=int, default=8192)
    parser.add_argument("--very-long-limit", type=int, default=16384)
    parser.add_argument(
        "--image-token-cost",
        type=int,
        default=1350,
        help="Fixed image-token allowance added to text tokens (default: 1350).",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete samples over --delete-limit unless protected by a keep pattern.",
    )
    parser.add_argument(
        "--keep-pattern",
        action="append",
        help="Filename glob protected from deletion; repeat as needed (default: *radar*, *step*).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search each dataset directory recursively.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def assistant_response(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return text_from_content(message.get("content"))
    for key in ("response", "answer"):
        value = text_from_content(record.get(key))
        if value:
            return value
    return ""


def percentage(value: int, total: int) -> str:
    return f"{value / total:.2%}" if total else "0.00%"


def json_files(folder: Path, recursive: bool) -> Iterable[Path]:
    return folder.rglob("*.json") if recursive else folder.glob("*.json")


def scan_folder(args: argparse.Namespace, tokenizer: Any, folder: Path) -> Stats:
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit(
            "tqdm is required. Run: pip install -r model/requirements.txt"
        ) from exc

    stats = Stats()
    long_examples: list[str] = []
    patterns = args.keep_pattern or ["*radar*", "*step*"]
    files = sorted(json_files(folder, args.recursive))

    for path in tqdm(files, desc=f"Scanning {folder.name}"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            response = assistant_response(record)
            if not response:
                continue
            text_tokens = len(tokenizer.encode(response, add_special_tokens=False))
            total_tokens = text_tokens + args.image_token_cost
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
            stats.errors += 1
            print(f"WARNING: cannot process {path}: {exc}")
            continue

        stats.valid += 1
        stats.maximum = max(stats.maximum, total_tokens)
        if total_tokens > args.report_limit:
            stats.over_report += 1
            if len(long_examples) < 5:
                long_examples.append(path.name)
        if total_tokens > args.delete_limit:
            stats.over_delete += 1
            protected = any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)
            if args.delete and not protected:
                try:
                    path.unlink()
                    stats.deleted += 1
                except OSError as exc:
                    stats.errors += 1
                    print(f"WARNING: cannot delete {path}: {exc}")
        if total_tokens > args.very_long_limit:
            stats.over_very_long += 1

    print(f"\n{folder}:")
    print(f"  valid samples: {stats.valid}")
    print(
        f"  > {args.report_limit}: {stats.over_report} "
        f"({percentage(stats.over_report, stats.valid)})"
    )
    print(
        f"  > {args.delete_limit}: {stats.over_delete} "
        f"({percentage(stats.over_delete, stats.valid)})"
    )
    print(
        f"  > {args.very_long_limit}: {stats.over_very_long} "
        f"({percentage(stats.over_very_long, stats.valid)})"
    )
    print(f"  maximum total tokens: {stats.maximum}")
    print(f"  deleted: {stats.deleted}; errors: {stats.errors}")
    if long_examples:
        print(f"  first long samples: {', '.join(long_examples)}")
    return stats


def main() -> None:
    args = parse_args()
    if not (0 <= args.report_limit <= args.delete_limit <= args.very_long_limit):
        raise SystemExit(
            "Expected 0 <= --report-limit <= --delete-limit <= --very-long-limit."
        )

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required. Run: pip install -r model/requirements.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    total = Stats()
    for folder in args.dataset_dir:
        if not folder.is_dir():
            print(f"WARNING: skipping missing directory: {folder}")
            total.errors += 1
            continue
        total.merge(scan_folder(args, tokenizer, folder))

    mode = "DELETE" if args.delete else "REPORT ONLY"
    print(f"\nSummary ({mode})")
    print(f"  valid samples: {total.valid}")
    print(f"  maximum total tokens: {total.maximum}")
    print(f"  > {args.report_limit}: {total.over_report}")
    print(f"  > {args.delete_limit}: {total.over_delete}")
    print(f"  > {args.very_long_limit}: {total.over_very_long}")
    print(f"  deleted: {total.deleted}; errors: {total.errors}")


if __name__ == "__main__":
    main()
