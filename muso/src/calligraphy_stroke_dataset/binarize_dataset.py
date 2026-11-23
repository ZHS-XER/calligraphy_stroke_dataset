import argparse
from pathlib import Path

from PIL import Image
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "data" / "dataset"

def resolve_path(path: Path | str) -> Path:

    path = Path(path)
    if not path.is_absolute():
        path = (Path(__file__).parent / path).resolve()
    return path

def binarize_image(img: Image.Image, threshold: int = 240) -> Image.Image:

    if img.mode != 'L':
        img = img.convert('L')

    img_array = np.array(img)

    binary_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)

    return Image.fromarray(binary_array, mode='L')

def process_dataset(
    dataset_dir: Path | str = DATASET_DIR,
    threshold: int = 240,
    dry_run: bool = False
) -> None:

    dataset_path = resolve_path(dataset_dir)

    if not dataset_path.exists():
        print(f"❌ Error: dataset directory does not exist - {dataset_path}")
        return

    char_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir()])

    total_images = 0
    processed_images = 0

    print(f"{'='*60}")
    print("🎨 Image binarization")
    print(f"{'='*60}")
    print(f"Dataset directory: {dataset_path}")
    print(f"Binarization threshold: {threshold}")
    print(f"Character count: {len(char_folders)}")

    if dry_run:
        print("⚠️  Preview mode (no files will be modified)")

    print(f"{'='*60}\n")

    for idx, char_folder in enumerate(char_folders, 1):

        image_files = sorted(char_folder.glob("stroke_*.png"))

        if idx % 100 == 0 or idx == 1:
            print(f"Progress: {idx}/{len(char_folders)} ({char_folder.name})")

        for img_file in image_files:
            total_images += 1

            try:

                img = Image.open(img_file)

                binary_img = binarize_image(img, threshold=threshold)

                if not dry_run:
                    binary_img.save(img_file)

                processed_images += 1

            except Exception as e:
                print(f"\n⚠️  Failed to process: {img_file}")
                print(f"   Error: {str(e)}")

    print(f"\n{'='*60}")
    print("✅ Processing complete")
    print(f"{'='*60}")
    print(f"Total images: {total_images}")
    print(f"Processed successfully: {processed_images}")

    if dry_run:
        print("\n💡 Preview done; run with --execute to apply changes")
    else:
        print("\n✅ All images binarized (gray→black, white preserved)")

    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Binarize all dataset images: turn gray pixels to black, keep white background"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=str(DATASET_DIR),
        help="Dataset directory (default: data/dataset, relative to repo root)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=int,
        default=240,
        help="Binarization threshold; values above are treated as white background (default: 240)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply changes (without this flag, run in preview)"
    )
    parser.add_argument(
        "--test",
        type=int,
        metavar="N",
        help="Test mode: process only the first N characters"
    )

    args = parser.parse_args()

    dataset_path = resolve_path(args.dataset)

    if args.test:
        print(f"\n⚠️  Test mode: processing only the first {args.test} characters\n")

        if not dataset_path.exists():
            print(f"❌ Error: dataset directory does not exist - {dataset_path}")
            return

        char_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir()])[:args.test]

        processed = 0
        for char_folder in char_folders:
            image_files = sorted(char_folder.glob("stroke_*.png"))
            print(f"Processing {char_folder.name}: {len(image_files)} images")

            for img_file in image_files:
                try:
                    img = Image.open(img_file)
                    binary_img = binarize_image(img, threshold=args.threshold)

                    if args.execute:
                        binary_img.save(img_file)

                    processed += 1
                except Exception as e:
                    print(f"  ❌ Failed: {img_file.name} - {str(e)}")

        print(f"\n✅ Test complete, processed {processed} images")
        if not args.execute:
            print("💡 Use --execute to apply actual changes")
    else:

        process_dataset(
            dataset_dir=dataset_path,
            threshold=args.threshold,
            dry_run=not args.execute
        )

if __name__ == "__main__":
    main()
