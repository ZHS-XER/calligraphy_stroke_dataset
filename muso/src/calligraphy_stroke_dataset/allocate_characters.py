import json
import shutil
from pathlib import Path
from typing import List, Dict

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
INTERMEDIATE_ROWS_DIR = DATA_DIR / "intermediate" / "grid_rows"
DATASET_DIR = DATA_DIR / "dataset"
CHAR_MAPPING_FILE = DATA_DIR / "metadata" / "character_strokes.json"

def load_character_mapping(mapping_file: Path | str = CHAR_MAPPING_FILE) -> List[Dict]:

    mapping_path = Path(mapping_file)
    if not mapping_path.is_absolute():
        mapping_path = (Path(__file__).parent / mapping_path).resolve()

    with open(mapping_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def allocate_characters_to_rows(
    row_dataset_dir: Path | str = INTERMEDIATE_ROWS_DIR,
    character_mapping_file: Path | str = CHAR_MAPPING_FILE,
    output_dir: Path | str = DATASET_DIR,
) -> None:

    print("=" * 60)
    print("🎯 Smart character allocation")
    print("=" * 60)

    print(f"\n1️⃣  Loading character mapping: {Path(character_mapping_file)}")
    char_mapping = load_character_mapping(character_mapping_file)
    print(f"   ✅ Loaded {len(char_mapping)} characters")

    print(f"\n2️⃣  Scanning row data: {Path(row_dataset_dir)}")
    row_path = Path(row_dataset_dir)
    if not row_path.is_absolute():
        row_path = (Path(__file__).parent / row_path).resolve()

    if not row_path.exists():
        print(f"   ❌ Error: dataset directory does not exist: {row_path}")
        return

    row_folders = sorted([d for d in row_path.iterdir() if d.is_dir() and d.name.startswith('row_')])
    print(f"   ✅ Found {len(row_folders)} rows")

    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = (Path(__file__).parent / output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\n3️⃣  Output directory: {output_path}")

    print(f"\n4️⃣  Start allocating characters...")
    print(f"{'='*60}")

    current_row_idx = 0

    success_count = 0
    skip_count = 0

    for char_info in char_mapping:
        char = char_info['character']
        cells_needed = char_info['cells_needed']
        stroke_count = char_info['stroke_count']
        strokes = char_info['strokes']
        rows_needed = char_info['rows_needed']

        char_dir = output_path / char
        char_dir.mkdir(exist_ok=True)

        collected_cells = []

        if current_row_idx + rows_needed > len(row_folders):
            print(f"   ⚠️  {char}: exceeds available rows (needs {rows_needed}, remaining {len(row_folders) - current_row_idx}), skipping")
            skip_count += 1
            current_row_idx += rows_needed
            continue

        cell_idx_in_char = 0
        for row_offset in range(rows_needed):
            row_folder = row_folders[current_row_idx + row_offset]

            cells_in_this_row = min(12, cells_needed - cell_idx_in_char)

            for cell_idx in range(cells_in_this_row):
                cell_file = row_folder / f"cell_{cell_idx:02d}.png"

                if cell_file.exists():
                    collected_cells.append(cell_file)
                else:
                    print(f"   ⚠️  {char}: missing cell {row_folder.name}/cell_{cell_idx:02d}.png")

                cell_idx_in_char += 1

        if len(collected_cells) == cells_needed:

            for idx, cell_file in enumerate(collected_cells):
                target_file = char_dir / f"stroke_{idx:02d}.png"
                shutil.copy2(cell_file, target_file)

            stroke_info = {
                "character": char,
                "stroke_count": stroke_count,
                "strokes": strokes,
                "total_images": len(collected_cells)
            }
            with open(char_dir / "stroke_info.json", 'w', encoding='utf-8') as f:
                json.dump(stroke_info, f, ensure_ascii=False, indent=2)

            if rows_needed > 1:
                print(f"   ✅ {char}: {stroke_count} strokes → {cells_needed} cells → spans {rows_needed} rows (row_{current_row_idx+1:04d}~{current_row_idx+rows_needed:04d})")
            else:
                print(f"   ✅ {char}: {stroke_count} strokes → {cells_needed} cells → row_{current_row_idx+1:04d}")

            success_count += 1
        else:
            print(f"   ❌ {char}: collected only {len(collected_cells)}/{cells_needed} cells")
            skip_count += 1

        current_row_idx += rows_needed

    print(f"{'='*60}")
    print(f"\n✅ Allocation complete")
    print(f"{'='*60}")
    print(f"Total characters: {len(char_mapping)}")
    print(f"Allocated successfully: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Output directory: {output_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    allocate_characters_to_rows()
