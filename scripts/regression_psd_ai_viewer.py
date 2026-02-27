#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


def _configure_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regression check for PSD/AI thumbnail and metadata decoding."
    )
    parser.add_argument(
        "--psd",
        action="append",
        default=[],
        help="Path to a real .psd sample file. Can be used multiple times.",
    )
    parser.add_argument(
        "--ai",
        action="append",
        default=[],
        help="Path to a real .ai sample file. Can be used multiple times.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=160,
        help="Thumbnail size to request (default: 160).",
    )
    return parser.parse_args()


def _ensure_qt_runtime() -> None:
    # Headless Linux/WSL runs need offscreen Qt.
    if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"


def _validate_path(path: Path, expected_suffix: str) -> str | None:
    if not path.exists():
        return f"{path}: file does not exist"
    if not path.is_file():
        return f"{path}: not a file"
    if path.suffix.lower() != expected_suffix:
        return f"{path}: expected {expected_suffix} extension"
    return None


def _normalize_cli_path(raw: str) -> Path:
    # Allow /mnt/c/... style paths when running with a Windows interpreter.
    if os.name == "nt":
        match = re.match(r"^/mnt/([a-zA-Z])/(.+)$", raw)
        if match:
            drive = match.group(1).upper()
            tail = match.group(2).replace("/", "\\")
            raw = f"{drive}:\\{tail}"
    return Path(raw).expanduser().resolve()


def main() -> int:
    _configure_import_path()
    args = _parse_args()
    _ensure_qt_runtime()

    from PyQt6.QtWidgets import QApplication

    from core.image_engine import ImageEngine

    psd_paths = [_normalize_cli_path(p) for p in args.psd]
    ai_paths = [_normalize_cli_path(p) for p in args.ai]
    if not psd_paths and not ai_paths:
        print("ERROR: provide at least one --psd or --ai file path")
        return 2

    failures: list[str] = []
    for path in psd_paths:
        issue = _validate_path(path, ".psd")
        if issue:
            failures.append(issue)
    for path in ai_paths:
        issue = _validate_path(path, ".ai")
        if issue:
            failures.append(issue)
    if failures:
        for issue in failures:
            print(f"FAIL: {issue}")
        return 2

    app = QApplication.instance() or QApplication([])
    _ = app  # keep QApplication alive for QPixmap operations
    engine = ImageEngine()

    test_matrix = [("PSD", path) for path in psd_paths] + [("AI", path) for path in ai_paths]
    failed = 0

    for kind, path in test_matrix:
        thumb = engine.load_thumbnail(str(path), args.size)
        meta = engine.load_metadata(str(path))

        thumb_broken = bool(thumb.metadata.get("broken"))
        thumb_null = thumb.pixmap.isNull()
        meta_broken = bool(meta.get("broken"))

        if thumb_broken or thumb_null or meta_broken:
            failed += 1
            print(
                "FAIL:",
                f"{kind} {path}",
                f"thumb_broken={thumb_broken}",
                f"thumb_null={thumb_null}",
                f"meta_broken={meta_broken}",
                f"thumb_source={thumb.metadata.get('source')}",
                f"meta_source={meta.get('source')}",
                f"thumb_error={thumb.metadata.get('error')}",
                f"meta_error={meta.get('error')}",
            )
            continue

        print(
            "PASS:",
            f"{kind} {path}",
            f"thumb_source={thumb.metadata.get('source')}",
            f"meta_source={meta.get('source')}",
        )

    print(f"Summary: {len(test_matrix) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
