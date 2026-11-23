from __future__ import annotations

import os
import argparse
import base64
import json
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
from tqdm import tqdm

API_URL = os.environ.get("MUSO_API_URL", "")
API_KEY = os.environ.get("MUSO_API_KEY", "")
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

DEFAULT_MODEL = "gemini-2.5-pro"
RETRY_DELAYS = [30, 40, 50, 60]
DEFAULT_IMAGE_BASE_NAME = "strokes_concat_except_last.png"
OUTPUT_DIR_BASE = "concat_input"

CALL_STATS = {
    "total_calls": 0,
    "successful_calls": 0,
    "failed_calls": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "start_time": None,
    "max_workers": 20,
    "model": None,
}

def get_retry_delay(attempt: int) -> float:

    if attempt < len(RETRY_DELAYS):
        return RETRY_DELAYS[attempt]
    return RETRY_DELAYS[-1]

def load_prompt_template() -> str:

    prompt_file = Path(__file__).parent / "prompt.md"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8").strip()

PROMPT_TEMPLATE = load_prompt_template()

def get_output_dir_name(model: str) -> str:

    model = (model or "").strip()
    if not model:
        return OUTPUT_DIR_BASE
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "_", model.lower()).strip("_")
    return f"{OUTPUT_DIR_BASE}_{normalized}" if normalized else OUTPUT_DIR_BASE

def image_to_base64(image_path: Path) -> str:

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def validate_llm_output(llm_output: Dict) -> bool:

    if not llm_output or not isinstance(llm_output, dict):
        return False
    if "answer" not in llm_output or "probabilities" not in llm_output:
        return False
    answer = llm_output.get("answer")
    if not answer or not isinstance(answer, str) or answer == "null":
        return False
    probabilities = llm_output.get("probabilities")
    if not probabilities or not isinstance(probabilities, list):
        return False
    first_prob = probabilities[0] if probabilities else None
    if not isinstance(first_prob, dict) or not first_prob.get("char"):
        return False
    return True

def safe_write_json(file_path: Path, data: Dict):

    tmp_path = file_path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        shutil.move(str(tmp_path), str(file_path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

def save_speed_test_results():

    script_dir = Path(__file__).parent
    result_file = script_dir / "speed_test_results.json"

    duration = time.time() - CALL_STATS["start_time"]
    total = CALL_STATS["total_calls"]
    success = CALL_STATS["successful_calls"]
    fail = CALL_STATS["failed_calls"]
    fail_rate = fail / total if total else 0.0
    speed = success / duration if duration > 0 else 0.0
    prompt_tokens = CALL_STATS["total_prompt_tokens"]
    completion_tokens = CALL_STATS["total_completion_tokens"]
    total_tokens = prompt_tokens + completion_tokens
    model_name = CALL_STATS.get("model")

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "parallelism": CALL_STATS["max_workers"],
        "total_calls": total,
        "success": success,
        "fail": fail,
        "fail_rate": round(fail_rate, 3),
        "duration_sec": round(duration, 1),
        "speed": round(speed, 2),
        "model": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    if result_file.exists():
        try:
            existing = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    else:
        existing = []

    existing.append(record)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    return record, result_file

def print_speed_stats():

    duration = time.time() - CALL_STATS["start_time"]
    total = CALL_STATS["total_calls"]
    success = CALL_STATS["successful_calls"]
    fail = CALL_STATS["failed_calls"]
    fail_rate = (fail / total * 100) if total else 0.0
    speed = success / duration if duration > 0 else 0.0
    prompt_tokens = CALL_STATS["total_prompt_tokens"]
    completion_tokens = CALL_STATS["total_completion_tokens"]
    total_tokens = prompt_tokens + completion_tokens
    model_name = CALL_STATS.get("model") or "N/A"

    print("\n" + "=" * 60)
    print("⚠️  Detected manual interruption; summarizing performance...")
    print("=" * 60)
    print(f"📈 Runtime: {duration:.1f} sec")
    print(f"📊 Total calls: {total}")
    print(f"✅ Success: {success}")
    print(f"❌ Failed: {fail}")
    print(f"🚫 Failure rate: {fail_rate:.1f}%")
    print(f"⚡ Avg speed: {speed:.2f} calls/sec")
    print(f"🧠 Model: {model_name}")
    print(f"🪙 Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
    print("=" * 60)

    try:
        record, result_file = save_speed_test_results()
        print(f"\n✅ Current results saved to {result_file.name}\n")
    except Exception as exc:
        print(f"\n❌ Failed to save results: {exc}\n")

def concat_image_name(base_name: str, drop_last: int) -> str:

    base_path = Path(base_name)
    stem = base_path.stem or "strokes_concat_except_last"
    suffix = base_path.suffix or ".png"
    if drop_last <= 1:
        return f"{stem}{suffix}" if base_path.suffix else f"{stem}{suffix}"
    return f"{stem}_{drop_last}{suffix}"

def is_result_valid(file_path: Path) -> bool:

    if file_path.suffix == ".tmp" or not file_path.exists():
        return False
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        llm_output = data.get("llm_output")
        return validate_llm_output(llm_output)
    except Exception:
        return False

def recognize_with_images(
    images: List[Path],
    character: str,
    remaining_strokes: int,
    *,
    max_retries: int = 300,
    initial_delay: float = 2.0,
    pbar: Optional[tqdm] = None,
    model: str = DEFAULT_MODEL,
) -> tuple[Optional[Dict], int]:

    try:
        image_contents = []
        for img_path in images:
            image_base64 = image_to_base64(img_path)
            image_contents.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                }
            )
    except Exception as exc:
        print(f"    ❌ Failed to read image: {str(exc)[:50]}")
        return None, 0

    prompt_with_context = PROMPT_TEMPLATE.replace("{x}", str(remaining_strokes))
    user_content = [{"type": "text", "text": prompt_with_context}] + image_contents

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个汉字识别助手"},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "temperature": 1.0,
    }

    for retry in range(max_retries):
        try:
            response = requests.post(
                API_URL, headers=HEADERS, json=data, timeout=120
            )
            if response.status_code == 200:
                result = response.json()
                usage = result.get("usage", {})

                def _safe_int(value) -> int:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return 0

                prompt_tokens = _safe_int(usage.get("prompt_tokens"))
                completion_tokens = _safe_int(usage.get("completion_tokens"))
                CALL_STATS["total_prompt_tokens"] += prompt_tokens
                CALL_STATS["total_completion_tokens"] += completion_tokens
                CALL_STATS["model"] = result.get("model") or model

                content = result["choices"][0]["message"]["content"].strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                try:
                    llm_output = json.loads(content)
                    if validate_llm_output(llm_output):
                        CALL_STATS["total_calls"] += 1
                        CALL_STATS["successful_calls"] += 1
                        if pbar:
                            pbar.update(1)
                        return llm_output, retry
                    CALL_STATS["total_calls"] += 1
                    CALL_STATS["failed_calls"] += 1
                except json.JSONDecodeError:
                    CALL_STATS["total_calls"] += 1
                    CALL_STATS["failed_calls"] += 1
            else:
                CALL_STATS["total_calls"] += 1
                CALL_STATS["failed_calls"] += 1

            if retry < max_retries - 1:
                wait_time = get_retry_delay(retry)
                time.sleep(wait_time)
        except requests.exceptions.Timeout:
            CALL_STATS["total_calls"] += 1
            CALL_STATS["failed_calls"] += 1
            if retry < max_retries - 1:
                wait_time = get_retry_delay(retry)
                time.sleep(wait_time)
        except Exception:
            CALL_STATS["total_calls"] += 1
            CALL_STATS["failed_calls"] += 1
            if retry < max_retries - 1:
                wait_time = get_retry_delay(retry)
                time.sleep(wait_time)

    return None, max_retries

def recognize_concat_variant(
    task: Dict,
    *,
    pbar: Optional[tqdm] = None,
    model: str = DEFAULT_MODEL,
) -> Dict:

    char_folder: Path = task["char_folder"]
    character: str = task["character"]
    drop_last: int = task["drop_last"]
    image_path: Path = task["image_path"]
    output_path: Path = task["output_path"]
    remaining_strokes: int = task["remaining_strokes"]

    output_path.parent.mkdir(exist_ok=True)

    if not image_path.exists():
        message = f"Concatenated image missing: {image_path.name}"
        print(f"  ⚠️ {character} remaining {drop_last} strokes: {message}")
        return {
            "character": character,
            "drop_last": drop_last,
            "status": "missing_image",
            "error": message,
            "retry_count": 0,
        }

    llm_output, retry_count = recognize_with_images(
        [image_path],
        character,
        remaining_strokes,
        pbar=pbar,
        model=model,
    )

    result_data = {
        "character": character,
        "drop_last": drop_last,
        "remaining_strokes": remaining_strokes,
        "image_name": image_path.name,
        "llm_output": llm_output,
        "retry_count": retry_count,
    }

    safe_write_json(output_path, result_data)
    status = "success" if llm_output else "failed"

    return {
        "character": character,
        "drop_last": drop_last,
        "status": status,
        "retry_count": retry_count,
        "output_file": output_path,
    }

def process_dataset(
    dataset_dir: str,
    image_base_name: str,
    drop_counts: Iterable[int],
    output_file: str,
    *,
    resume: bool = True,
    max_workers: int = 10,
    test_chars: Optional[int] = None,
    skip_chars: int = 0,
    model: str = DEFAULT_MODEL,
) -> List[Dict]:

    dataset_path = Path(dataset_dir).expanduser().resolve()
    script_dir = Path(__file__).parent
    output_path = script_dir / output_file
    drop_counts = sorted({d for d in drop_counts if d >= 1})
    output_dir_name = get_output_dir_name(model)

    if not dataset_path.exists():
        print(f"❌ Error: dataset directory does not exist - {dataset_path}")
        return []

    char_folders = sorted(
        [
            d
            for d in dataset_path.iterdir()
            if d.is_dir() and d.name and "\u4e00" <= d.name[0] <= "\u9fff"
        ],
        key=lambda p: p.name,
    )

    if skip_chars > 0:
        char_folders = char_folders[skip_chars:]
    if test_chars:
        char_folders = char_folders[:test_chars]

    print(f"\n{'='*60}")
    print("🎯 Concat input recognition (remove last 1/2/3 strokes)")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Characters: {len(char_folders)}")
    print(f"Parallelism: {max_workers}")
    print(f"Dropped strokes: {', '.join(str(d) for d in drop_counts)}")
    print(f"Output directory: {output_dir_name}/ under each character")

    tasks: List[Dict] = []
    char_to_tasks: Dict[str, List[int]] = {}

    for char_folder in char_folders:
        character = char_folder.name
        char_to_tasks[character] = []

        for drop_last in drop_counts:
            image_name = concat_image_name(image_base_name, drop_last)
            image_path = char_folder / image_name
            output_dir = char_folder / output_dir_name
            result_filename = Path(image_name).with_suffix(".json").name
            output_file_path = output_dir / result_filename

            if not image_path.exists():

                tasks.append(
                    {
                        "char_folder": char_folder,
                        "character": character,
                        "drop_last": drop_last,
                        "image_path": image_path,
                        "output_path": output_file_path,
                        "remaining_strokes": drop_last,
                    }
                )
                char_to_tasks[character].append(len(tasks) - 1)
                continue

            if resume and is_result_valid(output_file_path):
                continue

            tasks.append(
                {
                    "char_folder": char_folder,
                    "character": character,
                    "drop_last": drop_last,
                    "image_path": image_path,
                    "output_path": output_file_path,
                    "remaining_strokes": drop_last,
                }
            )
            char_to_tasks[character].append(len(tasks) - 1)

    total_api_calls = sum(1 for task in tasks if task["image_path"].exists())
    print(f"Tasks to process: {len(tasks)} (with {total_api_calls} API calls)")
    print(f"{'='*60}\n")

    if len(tasks) == 0:
        print("✅ All tasks have been processed!")
        return []

    CALL_STATS["start_time"] = time.time()
    CALL_STATS["max_workers"] = max_workers

    task_results: List[Optional[Dict]] = [None] * len(tasks)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=total_api_calls, desc="API calls", unit="calls", ncols=100) as pbar:
            future_to_idx = {}
            for idx, task in enumerate(tasks):
                if not task["image_path"].exists():
                    task_results[idx] = {
                        "character": task["character"],
                        "drop_last": task["drop_last"],
                        "status": "missing_image",
                    }
                    continue
                future = executor.submit(
                    recognize_concat_variant, task, pbar=pbar, model=model
                )
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    task_results[idx] = future.result()
                except Exception as exc:
                    print(f"\n❌ Task failed: {exc}")
                    task_results[idx] = {
                        "character": tasks[idx]["character"],
                        "drop_last": tasks[idx]["drop_last"],
                        "status": "exception",
                        "error": str(exc),
                    }

    total_time = time.time() - start_time

    summary = []
    for char_name, task_indices in char_to_tasks.items():
        entries = [
            task_results[i]
            for i in task_indices
            if i < len(task_results) and task_results[i] is not None
        ]
        summary.append(
            {
                "character": char_name,
                "results": entries,
            }
        )

    summary_sorted = sorted(summary, key=lambda item: item["character"])
    summary_data = {
        "total_characters": len(char_folders),
        "total_tasks": len(tasks),
        "api_calls": total_api_calls,
        "processing_time_seconds": total_time,
        "results": summary_sorted,
    }

    safe_write_json(output_path, summary_data)

    print(f"\n{'='*60}")
    print("✅ Done!")
    print(f"{'='*60}")
    print(f"Elapsed: {total_time:.2f} sec ({total_time/60:.1f} min)")
    print(f"Characters: {len(char_folders)}")
    print(f"Tasks: {len(tasks)} (API calls {total_api_calls})")
    model_name = CALL_STATS.get("model") or model
    prompt_tokens = CALL_STATS["total_prompt_tokens"]
    completion_tokens = CALL_STATS["total_completion_tokens"]
    total_tokens = prompt_tokens + completion_tokens
    print(f"Model: {model_name}")
    print(f"Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
    print(f"Summary file: {output_path}")
    print(f"Results: JSON files under {output_dir_name}/ for each character")
    print(f"{'='*60}\n")

    return summary_sorted

def main():
    parser = argparse.ArgumentParser(
        description="Concat stroke recognition: stitched images with last 1/2/3 strokes removed"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="../dataset",
        help="Dataset directory (default: ../dataset)",
    )
    parser.add_argument(
        "-b",
        "--base-name",
        type=str,
        default=DEFAULT_IMAGE_BASE_NAME,
        help=f"Base filename for stitched images (default: {DEFAULT_IMAGE_BASE_NAME})",
    )
    parser.add_argument(
        "--drop-counts",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="List of trailing-stroke counts to remove (default: 1 2 3)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="concat_summary.json",
        help="Summary output filename (default: concat_summary.json)",
    )
    parser.add_argument(
        "-w",
        "--max-workers",
        type=int,
        default=10,
        help="Number of parallel threads (default: 10)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume (always rerun all tasks)",
    )
    parser.add_argument(
        "--test",
        type=int,
        metavar="N",
        help="Process only first N characters (test mode)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        metavar="M",
        default=0,
        help="Skip first M characters",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name to call (default: {DEFAULT_MODEL})",
    )

    args = parser.parse_args()

    try:
        process_dataset(
            dataset_dir=args.dataset,
            image_base_name=args.base_name,
            drop_counts=args.drop_counts,
            output_file=args.output,
            resume=not args.no_resume,
            max_workers=args.max_workers,
            test_chars=args.test,
            skip_chars=args.skip,
            model=args.model,
        )
    except KeyboardInterrupt:
        print_speed_stats()
        sys.exit(0)

if __name__ == "__main__":
    main()
