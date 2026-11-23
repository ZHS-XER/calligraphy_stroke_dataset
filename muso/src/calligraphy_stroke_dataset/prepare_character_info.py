import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DEFAULT_HANZI_FILE = DATA_DIR / "inputs" / "hanzi.txt"
DEFAULT_BIHUA_FILE = DATA_DIR / "metadata" / "bihua.txt"
DEFAULT_OUTPUT_FILE = DATA_DIR / "metadata" / "character_strokes.json"

def load_bihua_dict(bihua_file: Path | str = DEFAULT_BIHUA_FILE) -> dict:

    bihua_dict = {}

    bihua_path = Path(bihua_file)
    if not bihua_path.is_absolute():
        bihua_path = (Path(__file__).parent / bihua_path).resolve()

    with open(bihua_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue

            char, strokes_str = line.split(':', 1)
            strokes = [s.strip() for s in strokes_str.split(',') if s.strip()]

            if char not in bihua_dict:
                bihua_dict[char] = strokes

    return bihua_dict

def read_hanzi_list(hanzi_file: Path | str) -> list:

    hanzi_path = Path(hanzi_file)
    if not hanzi_path.is_absolute():
        hanzi_path = (Path(__file__).parent / hanzi_path).resolve()

    with open(hanzi_path, 'r', encoding='utf-8') as f:
        content = f.read()

    hanzi_list = []
    for char in content:
        if '\u4e00' <= char <= '\u9fff':
            hanzi_list.append(char)

    return hanzi_list

def generate_character_strokes_mapping(
    hanzi_file: Path | str = DEFAULT_HANZI_FILE,
    bihua_file: Path | str = DEFAULT_BIHUA_FILE,
    output_file: Path | str = DEFAULT_OUTPUT_FILE,
) -> None:

    print("=" * 60)
    print("📖 Preparing Chinese character stroke info")
    print("=" * 60)

    print(f"\n1️⃣  Loading stroke database: {Path(bihua_file)}")
    bihua_dict = load_bihua_dict(bihua_file)
    print(f"   ✅ Loaded stroke info for {len(bihua_dict)} characters")

    print(f"\n2️⃣  Reading character list: {Path(hanzi_file)}")
    hanzi_list = read_hanzi_list(hanzi_file)
    print(f"   ✅ Loaded {len(hanzi_list)} characters")
    print(f"   First 10: {' '.join(hanzi_list[:10])}")

    print(f"\n3️⃣  Building character-to-strokes mapping...")
    character_mapping = []
    missing_chars = []

    for idx, char in enumerate(hanzi_list, 1):
        if char in bihua_dict:
            strokes = bihua_dict[char]
            stroke_count = len(strokes)
            cells_needed = stroke_count + 1

            rows_needed = (cells_needed + 11) // 12

            char_info = {
                "index": idx,
                "character": char,
                "stroke_count": stroke_count,
                "strokes": strokes,
                "cells_needed": cells_needed,
                "rows_needed": rows_needed
            }
            character_mapping.append(char_info)

            if rows_needed > 1:
                print(f"   📍 {char}: {stroke_count} strokes → needs {cells_needed} cells → spans {rows_needed} rows")
        else:
            print(f"   ⚠️  {char}: stroke info not found")
            missing_chars.append(char)

    print(f"\n4️⃣  Saving mapping file: {Path(output_file)}")
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = (Path(__file__).parent / output_path).resolve()

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(character_mapping, f, ensure_ascii=False, indent=2)

    print(f"\n" + "=" * 60)
    print(f"✅ Character-stroke mapping generated")
    print(f"=" * 60)
    print(f"Total characters: {len(hanzi_list)}")
    print(f"Mapped successfully: {len(character_mapping)}")
    print(f"Missing info: {len(missing_chars)}")
    if missing_chars:
        print(f"Missing characters: {', '.join(missing_chars)}")

    multi_row_chars = [c for c in character_mapping if c['rows_needed'] > 1]
    print(f"\nMulti-row characters: {len(multi_row_chars)}")
    for char_info in multi_row_chars[:10]:
        print(f"  {char_info['character']}: {char_info['stroke_count']} strokes → {char_info['rows_needed']} rows")
    if len(multi_row_chars) > 10:
        print(f"  ... and {len(multi_row_chars) - 10} more")

    print(f"\n💾 Mapping file saved: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    generate_character_strokes_mapping()
