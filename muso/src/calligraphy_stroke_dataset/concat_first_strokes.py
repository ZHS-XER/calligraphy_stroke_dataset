from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "data" / "dataset"

STROKE_PATTERN = re.compile(r"stroke_(\d+)\.png$")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate the first strokes for each character; optionally drop trailing strokes."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Root directory of the character dataset (default: data/dataset)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="strokes_concat_except_last.png",
        help="Output filename when dropping only the last 1 stroke; other drop counts append '_{drop}'.",
    )
    parser.add_argument(
        "--drop-counts",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="List of trailing stroke counts to drop (default: 1 2 3).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing target images if present.",
    )
    return parser.parse_args()

def collect_strokes(char_dir: Path) -> List[Tuple[int, Path]]:

    strokes: List[Tuple[int, Path]] = []
    for img_path in char_dir.glob("stroke_*.png"):
        match = STROKE_PATTERN.match(img_path.name)
        if not match:
            continue
        strokes.append((int(match.group(1)), img_path))
    strokes.sort(key=lambda item: item[0])
    return strokes

def select_target_strokes(strokes: List[Tuple[int, Path]], drop_last: int) -> List[Path]:

    if not strokes:
        return []

    max_index = strokes[-1][0]
    target_max = max_index - drop_last
    if target_max <= 0:
        return []

    selected: List[Path] = []
    for idx, path in strokes:
        if idx == 0:
            continue
        if idx > target_max:
            continue
        selected.append(path)
    return selected

def load_images(paths: Iterable[Path]) -> List[Image.Image]:

    images: List[Image.Image] = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.convert("RGBA"))
    return images

def concat_images_horizontally(images: List[Image.Image]) -> Image.Image:

    if not images:
        raise ValueError("No images available for concatenation.")

    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    canvas = Image.new("RGBA", (total_width, max_height), (255, 255, 255, 0))

    offset = 0
    for img in images:
        canvas.paste(img, (offset, 0), mask=img if img.mode == "RGBA" else None)
        offset += img.width

    return canvas

def build_output_name(base_name: str, drop_last: int) -> str:

    if drop_last == 1:
        return base_name
    base_path = Path(base_name)
    stem = base_path.stem or "strokes_concat_except_last"
    suffix = base_path.suffix or ".png"
    return f"{stem}_{drop_last}{suffix}"

def process_character_dir(
    char_dir: Path,
    output_name: str,
    drop_counts: Iterable[int],
    overwrite: bool,
) -> List[Tuple[int, bool, str]]:
    strokes = collect_strokes(char_dir)
    results: List[Tuple[int, bool, str]] = []

    for drop_last in sorted({d for d in drop_counts if d >= 1}):
        selected_paths = select_target_strokes(strokes, drop_last)
        if not selected_paths:
            results.append(
                (
                    drop_last,
                    False,
                    f"Insufficient strokes after dropping {drop_last}; need at least 1 remaining.",
                )
            )
            continue

        output_filename = build_output_name(output_name, drop_last)
        output_path = char_dir / output_filename
        if output_path.exists() and not overwrite:
            results.append(
                (
                    drop_last,
                    False,
                    f"{output_filename} already exists; use --overwrite to replace.",
                )
            )
            continue

        images = load_images(selected_paths)
        result = concat_images_horizontally(images)
        result.save(output_path)
        results.append((drop_last, True, f"Generated {output_filename}"))

    return results

def main() -> None:
    args = parse_args()
    dataset_dir: Path = args.dataset_dir
    if not dataset_dir.is_absolute():
        dataset_dir = (ROOT_DIR / dataset_dir).resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    processed = 0
    skipped = 0

    for char_dir in sorted(dataset_dir.iterdir()):
        if not char_dir.is_dir():
            continue
        results = process_character_dir(
            char_dir,
            args.output_name,
            args.drop_counts,
            args.overwrite,
        )
        print(char_dir.name + ":")
        for drop_last, success, message in results:
            status = "✔" if success else "✗"
            print(f"  [{status}] Drop last {drop_last} strokes -> {message}")
            if success:
                processed += 1
            else:
                skipped += 1

    print(f"\nDone: created {processed} concatenated images, skipped {skipped} tasks.")

if __name__ == "__main__":
    main()
