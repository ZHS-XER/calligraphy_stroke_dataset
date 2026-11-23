import os
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DATASET_DIR = DATA_DIR / "dataset"
OUTPUT_FILE = ROOT_DIR / "stroke_analysis_results.json"

def resolve_path(path: Path | str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = (Path(__file__).parent / path).resolve()
    return path

API_URL = os.environ.get("MUSO_API_URL", "")
API_KEY = os.environ.get("MUSO_API_KEY", "")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}
RETRY_DELAYS = [30, 40, 50, 60]

def get_retry_delay(attempt: int) -> float:

    if attempt < len(RETRY_DELAYS):
        return RETRY_DELAYS[attempt]
    return RETRY_DELAYS[-1]

def recognize_stroke_info(character: str, max_retries: int = 15, initial_delay: float = 0.5) -> Optional[Dict]:

    prompt = f"""请精确分析汉字"{character}"的笔画。

要求：
1. 严格按照标准楷书笔顺统计笔画数
2. 使用准确的笔画术语

输出格式（严格JSON）：
{{
  "character": "{character}",
  "stroke_count": <数字>,
  "strokes": ["笔画1", "笔画2", ...]
}}

现在请分析汉字"{character}"并输出JSON。"""

    data = {
        "model": "gemini-2.5-pro",
        "messages": [
            {
                "role": "system",
                "content": "你是一个汉字笔画分析专家，精通汉字的笔画和笔顺。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "temperature": 1.0
    }

    for retry in range(max_retries):
        try:
            response = requests.post(API_URL, headers=HEADERS, json=data, timeout=30)

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()

                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                try:
                    stroke_data = json.loads(content)

                    required_fields = ['character', 'stroke_count', 'strokes']
                    if all(field in stroke_data for field in required_fields):
                        stroke_data['success'] = True
                        return stroke_data
                    else:
                        print("    Warning: JSON missing required fields")
                        if retry < max_retries - 1:
                            wait_time = get_retry_delay(retry)
                            time.sleep(wait_time)
                            continue
                        return None

                except json.JSONDecodeError as e:
                    print(f"    Warning: JSON parse failed - {str(e)[:50]}")
                    if retry < max_retries - 1:
                        wait_time = get_retry_delay(retry)
                        time.sleep(wait_time)
                        continue
                    return None
            else:
                print(f"    Retry {retry + 1}/{max_retries}: status code {response.status_code}")
                wait_time = get_retry_delay(retry)
                time.sleep(wait_time)

        except requests.exceptions.Timeout:
            print(f"    Retry {retry + 1}/{max_retries}: request timeout")
            wait_time = get_retry_delay(retry)
            time.sleep(wait_time)
        except Exception as e:
            print(f"    Retry {retry + 1}/{max_retries}: error - {str(e)[:50]}")
            wait_time = get_retry_delay(retry)
            time.sleep(wait_time)

    print("    ❌ All retries failed")
    return None

def analyze_character_simple(character: str) -> Optional[Dict]:

    print(f"Analyzing character: '{character}'", end=" ", flush=True)

    start_time = time.time()

    result = recognize_stroke_info(character, max_retries=20, initial_delay=0.5)

    elapsed_time = time.time() - start_time

    if result and result.get('success'):
        print(f"✅ (elapsed: {elapsed_time:.1f}s)")
        print(f"  Strokes: {result['stroke_count']}, strokes list: {', '.join(result['strokes'][:5])}{'...' if len(result['strokes']) > 5 else ''}")
        return result
    else:
        print(f"❌ Failed (elapsed: {elapsed_time:.1f}s)")
        return None

def analyze_single_char_wrapper(char_data: tuple) -> Dict:

    idx, total, character, char_folder = char_data

    print(f"[{idx}/{total}] ", end="", flush=True)

    stroke_info = analyze_character_simple(character)

    if stroke_info:

        images = list(char_folder.glob("*.png"))

        result = {
            'character': character,
            'stroke_count': stroke_info['stroke_count'],
            'strokes': stroke_info['strokes'],
            'image_count': len(images)
        }
        return result
    else:
        return {
            'character': character,
            'error': 'Analysis failed'
        }

def analyze_dataset_characters(
    dataset_dir: Path | str = DATASET_DIR,
    max_workers: int = 3,
    save_interval: int = 10,
    resume: bool = True
) -> List[Dict]:

    dataset_path = resolve_path(dataset_dir)
    output_file = OUTPUT_FILE

    if not dataset_path.exists():
        print(f"❌ Error: dataset directory does not exist - {dataset_path}")
        return []

    char_folders = sorted([
        d for d in dataset_path.iterdir()
        if d.is_dir() and len(d.name) > 0
    ])

    char_folders = [d for d in char_folders if '\u4e00' <= d.name[0] <= '\u9fff']
    all_characters = [d.name for d in char_folders]

    existing_results = {}
    if resume and output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                saved_results = json.load(f)
                for result in saved_results:
                    existing_results[result['character']] = result
            print(f"📂 Found existing results file; already completed {len(existing_results)} characters")
        except:
            print("⚠️  Failed to read existing results file; restarting from scratch")

    remaining_folders = [
        d for d in char_folders
        if d.name not in existing_results
    ]

    print(f"\n{'='*60}")
    print(f"Total characters in dataset: {len(char_folders)}")
    print(f"Completed: {len(existing_results)}")
    print(f"Remaining: {len(remaining_folders)}")
    print(f"Parallel workers: {max_workers}, max 15 retries per character on failure")
    print(f"{'='*60}\n")

    if len(remaining_folders) == 0:
        print("✅ All characters have been processed!")
        return list(existing_results.values())

    results = list(existing_results.values())
    start_time = time.time()

    total_chars = len(char_folders)
    already_done = len(existing_results)
    tasks = [
        (already_done + idx, total_chars, char_folder.name, char_folder)
        for idx, char_folder in enumerate(remaining_folders, 1)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        future_to_idx = {
            executor.submit(analyze_single_char_wrapper, task): task[0]
            for task in tasks
        }

        completed_count = already_done
        for future in as_completed(future_to_idx):
            result = future.result()
            results.append(result)
            completed_count += 1

            if (completed_count - already_done) % save_interval == 0 or completed_count == total_chars:

                results_sorted = sorted(results, key=lambda x: x['character'])
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results_sorted, f, ensure_ascii=False, indent=2)
                print(f"\n  💾 Saved {completed_count}/{total_chars} results to {output_file}")
                print("-" * 60)

    total_time = time.time() - start_time
    success_count = sum(1 for r in results if 'stroke_count' in r)

    results_sorted = sorted(results, key=lambda x: x['character'])
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_sorted, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("✅ Finished!")
    print(f"  Elapsed: {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"  Processed this run: {len(remaining_folders)} characters")
    print(f"  Total completed: {len(results)}/{total_chars}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {len(results) - success_count}")
    if len(remaining_folders) > 0:
        print(f"  Avg per character: {total_time/len(remaining_folders):.2f}s")
    print(f"  Results saved to: {output_file}")
    print(f"{'='*60}\n")

    return results_sorted

def print_summary(results: List[Dict]):

    if not results:
        print("No results to display")
        return

    successful_results = [r for r in results if 'stroke_count' in r]

    if not successful_results:
        print("No successful analysis results")
        return

    print(f"\n{'='*60}")
    print("Stroke count summary")
    print(f"{'='*60}\n")

    stroke_groups = {}
    for r in successful_results:
        count = r['stroke_count']
        if count not in stroke_groups:
            stroke_groups[count] = []
        stroke_groups[count].append(r['character'])

    for stroke_count in sorted(stroke_groups.keys()):
        chars = stroke_groups[stroke_count]
        print(f"{stroke_count:2d} strokes ({len(chars):2d} chars): {' '.join(chars[:20])}")
        if len(chars) > 20:
            print(f"      ... plus {len(chars)} total")

    print(f"\nStroke count range: {min(stroke_groups.keys())} - {max(stroke_groups.keys())}")
    print(f"Average stroke count: {sum(r['stroke_count'] for r in successful_results) / len(successful_results):.1f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Chinese character stroke information")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=str(DATASET_DIR),
        help="Dataset directory (default: data/dataset, relative to repo root)"
    )
    parser.add_argument(
        "-w",
        "--max-workers",
        type=int,
        default=3,
        help="Number of characters processed in parallel (default: 3)"
    )
    parser.add_argument(
        "-s",
        "--save-interval",
        type=int,
        default=10,
        help="Save interval (default: every 10 characters)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume; start over"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: analyze only the first 5 characters"
    )

    args = parser.parse_args()
    dataset_path = resolve_path(args.dataset)

    if args.test:
        print("\n⚠️  Test mode: analyze only the first 5 characters\n")

        if not dataset_path.exists():
            print(f"❌ Error: dataset directory does not exist - {dataset_path}")
            exit(1)

        char_folders = sorted([
            d for d in dataset_path.iterdir()
            if d.is_dir() and len(d.name) > 0 and '\u4e00' <= d.name[0] <= '\u9fff'
        ])[:5]

        tasks = [
            (idx, 5, char_folder.name, char_folder)
            for idx, char_folder in enumerate(char_folders, 1)
        ]

        results = []
        with ThreadPoolExecutor(max_workers=min(args.max_workers, 5)) as executor:
            future_to_idx = {
                executor.submit(analyze_single_char_wrapper, task): task[0]
                for task in tasks
            }

            for future in as_completed(future_to_idx):
                result = future.result()
                results.append(result)

        results_sorted = sorted(results, key=lambda x: x['character'])
        test_output = ROOT_DIR / "stroke_analysis_test.json"
        with open(test_output, 'w', encoding='utf-8') as f:
            json.dump(results_sorted, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Test results saved to: {test_output}")

        print_summary(results_sorted)
    else:

        results = analyze_dataset_characters(
            dataset_dir=dataset_path,
            max_workers=args.max_workers,
            save_interval=args.save_interval,
            resume=not args.no_resume
        )

        print_summary(results)
