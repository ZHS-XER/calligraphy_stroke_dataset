import os
import requests
import base64
import json
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from typing import Dict, List, Optional
import argparse
import re
from tqdm import tqdm
from datetime import datetime

API_URL = os.environ.get("MUSO_API_URL", "")
API_KEY = os.environ.get("MUSO_API_KEY", "")
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

DEFAULT_MODEL = "gemini-2.5-pro"
RETRY_DELAYS = [30, 40, 50, 60]

CALL_STATS = {
    "total_calls": 0,
    "successful_calls": 0,
    "failed_calls": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "start_time": None,
    "max_workers": 20,
    "model": None
}

def get_retry_delay(attempt: int) -> float:

    if attempt < len(RETRY_DELAYS):
        return RETRY_DELAYS[attempt]
    return RETRY_DELAYS[-1]

def load_prompt_template() -> str:

    prompt_file = Path(__file__).parent / "prompt.md"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()

PROMPT_TEMPLATE = load_prompt_template()

OUTPUT_DIR_BASE = "COF_input"

def get_output_dir_name(model: str) -> str:

    model = (model or "").strip()
    if not model:
        return OUTPUT_DIR_BASE
    normalized = re.sub(r'[^a-zA-Z0-9._-]+', '_', model.lower())
    normalized = normalized.strip("_")
    if not normalized:
        return OUTPUT_DIR_BASE
    return f"{OUTPUT_DIR_BASE}_{normalized}"

def image_to_base64(image_path: Path) -> str:

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def validate_llm_output(llm_output: Dict) -> bool:

    if not llm_output or not isinstance(llm_output, dict):
        return False

    if 'answer' not in llm_output or 'probabilities' not in llm_output:
        return False

    answer = llm_output.get('answer')
    if not answer or answer == 'null' or not isinstance(answer, str):
        return False

    probabilities = llm_output.get('probabilities')
    if not probabilities or not isinstance(probabilities, list) or len(probabilities) == 0:
        return False

    first_prob = probabilities[0]
    if not isinstance(first_prob, dict):
        return False

    if 'char' not in first_prob or not first_prob.get('char'):
        return False

    return True

def safe_write_json(file_path: Path, data: Dict):

    tmp_path = file_path.with_suffix('.tmp')

    try:

        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        shutil.move(str(tmp_path), str(file_path))
    except Exception as e:

        if tmp_path.exists():
            tmp_path.unlink()
        raise e

def is_stroke_file_valid(file_path: Path, expected_runs: int = 1) -> bool:

    if file_path.suffix == '.tmp':
        return False

    if not file_path.exists():
        return False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        llm_output = data.get('llm_output')
        if not validate_llm_output(llm_output):
            return False

        return True
    except Exception:
        return False

def save_speed_test_results():

    script_dir = Path(__file__).parent
    result_file = script_dir / "speed_test_results.json"

    duration = time.time() - CALL_STATS["start_time"]
    total = CALL_STATS["total_calls"]
    success = CALL_STATS["successful_calls"]
    fail = CALL_STATS["failed_calls"]
    fail_rate = fail / total if total > 0 else 0.0
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
        "total_tokens": total_tokens
    }

    if result_file.exists():
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            data = []
    else:
        data = []

    data.append(record)

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return record, result_file

def print_speed_stats():

    duration = time.time() - CALL_STATS["start_time"]
    total = CALL_STATS["total_calls"]
    success = CALL_STATS["successful_calls"]
    fail = CALL_STATS["failed_calls"]
    fail_rate = (fail / total * 100) if total > 0 else 0.0
    speed = success / duration if duration > 0 else 0.0
    prompt_tokens = CALL_STATS["total_prompt_tokens"]
    completion_tokens = CALL_STATS["total_completion_tokens"]
    total_tokens = prompt_tokens + completion_tokens
    model_name = CALL_STATS.get("model") or "N/A"

    print("\n" + "="*60)
    print("⚠️  Detected manual interruption; summarizing performance...")
    print("="*60)
    print(f"📈 Runtime: {duration:.1f} sec")
    print(f"📊 Total calls: {total}")
    print(f"✅ Success: {success}")
    print(f"❌ Failed: {fail}")
    print(f"🚫 Failure rate: {fail_rate:.1f}%")
    print(f"⚡ Avg speed: {speed:.2f} calls/sec")
    print(f"🧠 Model: {model_name}")
    print(f"🪙 Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
    print("="*60)

    try:
        record, result_file = save_speed_test_results()
        print(f"\n✅ Current results saved to {result_file.name}\n")
    except Exception as e:
        print(f"\n❌ Failed to save results: {str(e)}\n")

def select_frame_images(char_folder: Path, target_stroke: int, total_strokes: int, num_pre_frames: int) -> List[Path]:

    pre_range = list(range(1, target_stroke))

    selected_images = []

    if len(pre_range) == 0:

        pass
    elif len(pre_range) <= num_pre_frames:

        for idx in pre_range:
            img_path = char_folder / f"stroke_{idx:02d}.png"
            if img_path.exists():
                selected_images.append(img_path)
    else:

        step = (len(pre_range) - 1) / (num_pre_frames - 1) if num_pre_frames > 1 else 0
        selected_indices = []
        for i in range(num_pre_frames):
            idx = pre_range[round(i * step)]
            selected_indices.append(idx)

        selected_indices = sorted(list(set(selected_indices)))

        for idx in selected_indices:
            img_path = char_folder / f"stroke_{idx:02d}.png"
            if img_path.exists():
                selected_images.append(img_path)

    target_img_path = char_folder / f"stroke_{target_stroke:02d}.png"
    if target_img_path.exists():
        selected_images.append(target_img_path)

    return selected_images

def recognize_with_frames(images: List[Path], character: str, remaining_strokes: int,
                         max_retries: int = 300, initial_delay: float = 2.0,
                         pbar: Optional[tqdm] = None,
                         model: str = DEFAULT_MODEL) -> tuple[Optional[Dict], int]:

    try:

        image_contents = []
        for img_path in images:
            image_base64 = image_to_base64(img_path)
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            })
    except Exception as e:
        print(f"    ❌ Failed to read image: {str(e)[:50]}")
        return None, 0

    prompt_with_context = PROMPT_TEMPLATE.replace("{x}", str(remaining_strokes))

    user_content = [{"type": "text", "text": prompt_with_context}] + image_contents

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是一个汉字识别助手"
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        "stream": False,
        "temperature": 1.0
    }

    for retry in range(max_retries):
        try:
            response = requests.post(API_URL, headers=HEADERS, json=data, timeout=120)

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
                    else:

                        CALL_STATS["total_calls"] += 1
                        CALL_STATS["failed_calls"] += 1
                        if retry < max_retries - 1:
                            wait_time = get_retry_delay(retry)
                            time.sleep(wait_time)
                            continue
                        return None, retry + 1

                except json.JSONDecodeError as e:

                    CALL_STATS["total_calls"] += 1
                    CALL_STATS["failed_calls"] += 1
                    if retry < max_retries - 1:
                        wait_time = get_retry_delay(retry)
                        time.sleep(wait_time)
                        continue
                    return None, retry + 1
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
        except Exception as e:

            CALL_STATS["total_calls"] += 1
            CALL_STATS["failed_calls"] += 1
            if retry < max_retries - 1:
                wait_time = get_retry_delay(retry)
                time.sleep(wait_time)

    return None, max_retries

def recognize_single_stroke(char_folder: Path, character: str,
                          target_stroke: int, total_strokes: int,
                          chain_length: int,
                          pbar: Optional[tqdm] = None,
                          model: str = DEFAULT_MODEL) -> Dict:

    stroke_name = f"stroke_{target_stroke:02d}"

    remaining_strokes = max(total_strokes - target_stroke, 0)

    output_dir_name = get_output_dir_name(model)
    output_dir = char_folder / output_dir_name
    output_dir.mkdir(exist_ok=True)

    try:
        input_images = select_frame_images(char_folder, target_stroke, total_strokes, chain_length)
    except Exception as e:
        print(f"  ⚠️ {character} {stroke_name}_L{chain_length}: failed to select image - {str(e)}")
        return {
            'character': character,
            'stroke_name': stroke_name,
            'chain_length': chain_length,
            'status': 'image_selection_failed',
            'error': str(e)
        }

    if not input_images:
        print(f"  ⚠️ {character} {stroke_name}_L{chain_length}: no available images")
        return {
            'character': character,
            'stroke_name': stroke_name,
            'chain_length': chain_length,
            'status': 'no_images'
        }

    llm_output, retry_count = recognize_with_frames(
        input_images, character, remaining_strokes, pbar=pbar, model=model
    )

    output_data = {
        'character': character,
        'stroke_name': stroke_name,
        'target_stroke': target_stroke,
        'remaining_strokes': remaining_strokes,
        'total_strokes': total_strokes,
        'chain_length': chain_length,
        'input_frames': [img.name for img in input_images],
        'llm_output': llm_output,
        'retry_count': retry_count
    }

    output_file = output_dir / f"{stroke_name}_L{chain_length}.json"
    safe_write_json(output_file, output_data)

    retry_stats = {
        f"{stroke_name}_L{chain_length}": [
            {'run_id': 1, 'retry_count': retry_count}
        ]
    }

    success_count = 1 if llm_output is not None else 0

    return {
        'character': character,
        'stroke_name': stroke_name,
        'chain_length': chain_length,
        'target_stroke': target_stroke,
        'success_count': success_count,
        'retry_stats': retry_stats,
        'status': 'completed'
    }

def recognize_character(char_folder: Path, character: str,
                       pbar: Optional[tqdm] = None) -> Dict:

    stroke_info_file = char_folder / "stroke_info.json"
    if not stroke_info_file.exists():
        print(f"  ⚠️ {character}: stroke_info.json not found")
        return {
            'character': character,
            'status': 'missing_stroke_info'
        }

    try:
        with open(stroke_info_file, 'r', encoding='utf-8') as f:
            stroke_info = json.load(f)
    except Exception as e:
        print(f"  ⚠️ {character}: failed to read stroke_info.json: {str(e)}")
        return {
            'character': character,
            'status': 'invalid_stroke_info'
        }

    stroke_count = stroke_info.get('stroke_count', len(stroke_info.get('strokes', [])))

    if stroke_count < 3:
        print(f"  ⚠️ {character}: insufficient strokes ({stroke_count}, need at least 3)")
        return {
            'character': character,
            'status': 'insufficient_strokes',
            'stroke_count': stroke_count
        }

    output_dir_name = get_output_dir_name(model)
    output_dir = char_folder / output_dir_name
    output_dir.mkdir(exist_ok=True)

    target_strokes = []
    for offset in [1, 2, 3]:
        target_idx = stroke_count - offset
        if target_idx >= 1:
            target_strokes.append(target_idx)

    if not target_strokes:
        return {
            'character': character,
            'status': 'insufficient_targets',
            'stroke_count': stroke_count
        }

    retry_stats = {}
    tasks_processed = 0

    for target_stroke in target_strokes:
        for chain_length in [2, 3, 4]:
            result = recognize_single_stroke(
                char_folder,
                character,
                target_stroke,
                stroke_count,
                chain_length,
                pbar
            )
            tasks_processed += 1
            if result and 'retry_stats' in result:
                retry_stats.update(result['retry_stats'])

    return {
        'character': character,
        'status': 'completed',
        'chains_processed': tasks_processed,
        'total_strokes': stroke_count,
        'target_strokes': target_strokes,
        'retry_stats': retry_stats
    }

def process_dataset(dataset_dir: str = "../dataset",
                   output_file: str = "cof_summary.json",
                   resume: bool = True,
                   max_workers: int = 20,
                   test_chars: Optional[int] = None,
                   skip_chars: int = 0,
                   model: str = DEFAULT_MODEL) -> List[Dict]:

    CALL_STATS["start_time"] = time.time()
    CALL_STATS["max_workers"] = max_workers
    CALL_STATS["total_calls"] = 0
    CALL_STATS["successful_calls"] = 0
    CALL_STATS["failed_calls"] = 0
    CALL_STATS["total_prompt_tokens"] = 0
    CALL_STATS["total_completion_tokens"] = 0
    CALL_STATS["model"] = model

    script_dir = Path(__file__).parent
    dataset_path = script_dir / dataset_dir
    output_path = script_dir / output_file
    output_dir_name = get_output_dir_name(model)

    if not dataset_path.exists():
        print(f"❌ Error: dataset directory does not exist - {dataset_path}")
        return []

    char_folders = sorted([
        d for d in dataset_path.iterdir()
        if d.is_dir() and len(d.name) > 0 and '\u4e00' <= d.name[0] <= '\u9fff'
    ])

    if skip_chars > 0:
        char_folders = char_folders[skip_chars:]
    if test_chars:
        char_folders = char_folders[:test_chars]

    print(f"\n{'='*60}")
    print(f"🎯 Chain-of-Frames chain-length experiment (last 1/2/3 strokes)")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Characters: {len(char_folders)}")
    print(f"Parallelism: {max_workers}")
    print(f"Strategy: handle last 1st/2nd/3rd strokes in parallel")
    print(f"Chain lengths: L=2, L=3, L=4 (preceding frames)")
    print(f"Runs per chain length: 1")
    print(f"Output location: {output_dir_name}/ folder under each character")

    retry_stats_path = script_dir / output_file.replace('.json', '_retry.json')

    all_retry_stats = {}
    if resume and retry_stats_path.exists():
        try:
            with open(retry_stats_path, 'r', encoding='utf-8') as f:
                all_retry_stats = json.load(f)
        except:
            pass

    stroke_tasks = []
    char_to_tasks = {}

    for char_folder in char_folders:
        character = char_folder.name

        stroke_info_file = char_folder / "stroke_info.json"
        if not stroke_info_file.exists():
            continue

        try:
            with open(stroke_info_file, 'r', encoding='utf-8') as f:
                stroke_info = json.load(f)
            stroke_count = stroke_info.get('stroke_count', len(stroke_info.get('strokes', [])))
        except:
            continue

        output_dir = char_folder / output_dir_name
        output_dir.mkdir(exist_ok=True)

        char_to_tasks[character] = []

        target_strokes = [stroke_count - 1, stroke_count - 2, stroke_count - 3]
        valid_targets = [idx for idx in target_strokes if idx >= 1]
        if not valid_targets:
            continue

        for target_stroke in valid_targets:
            for chain_length in [2, 3, 4]:
                stroke_file = output_dir / f"stroke_{target_stroke:02d}_L{chain_length}.json"
                if resume and is_stroke_file_valid(stroke_file, expected_runs=1):
                    continue

                task_idx = len(stroke_tasks)
                stroke_tasks.append({
                    'char_folder': char_folder,
                    'character': character,
                    'target_stroke': target_stroke,
                    'total_strokes': stroke_count,
                    'chain_length': chain_length
                })
                char_to_tasks[character].append(task_idx)

    print(f"Tasks to process: {len(stroke_tasks)} (character × stroke × chain length)")
    print(f"{'='*60}\n")

    if len(stroke_tasks) == 0:
        print("✅ All tasks have been processed!")
        return []

    total_api_calls = len(stroke_tasks)

    print(f"Estimated API calls: {total_api_calls}")
    print(f"{'='*60}\n")

    results = []
    start_time = time.time()

    task_results = [None] * len(stroke_tasks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=total_api_calls, desc="API calls", unit="calls",
                  ncols=100, smoothing=0.1) as pbar:
            future_to_idx = {
                executor.submit(
                    recognize_single_stroke,
                    task['char_folder'],
                    task['character'],
                    task['target_stroke'],
                    task['total_strokes'],
                    task['chain_length'],
                    pbar,
                    model
                ): idx
                for idx, task in enumerate(stroke_tasks)
            }

            completed_tasks = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    task_results[idx] = result

                    if result and 'retry_stats' in result:
                        char_name = result['character']
                        if char_name not in all_retry_stats:
                            all_retry_stats[char_name] = {}
                        all_retry_stats[char_name].update(result['retry_stats'])

                    completed_tasks += 1

                    if completed_tasks % 10 == 0:

                        char_summary = {}
                        for char_name, task_indices in char_to_tasks.items():
                            tasks_processed = sum(
                                1 for ti in task_indices
                                if ti < len(task_results) and task_results[ti] is not None
                            )
                            if tasks_processed > 0 or len(task_indices) > 0:
                                char_summary[char_name] = tasks_processed

                        results_sorted = [
                            {'character': char, 'tasks_processed': count}
                            for char, count in sorted(char_summary.items())
                        ]

                        summary_data = {
                            'total_characters': len(char_folders),
                            'completed_tasks': completed_tasks,
                            'total_tasks': len(stroke_tasks),
                            'processing_time_seconds': time.time() - start_time,
                            'results': results_sorted
                        }
                        safe_write_json(output_path, summary_data)

                        if all_retry_stats:
                            safe_write_json(retry_stats_path, all_retry_stats)

                        pbar.set_postfix_str(f"Completed {completed_tasks}/{len(stroke_tasks)} tasks")

                except Exception as e:
                    print(f"\n❌ Task processing failed: {str(e)[:100]}")
                    task_results[idx] = None

    total_time = time.time() - start_time

    char_summary = {}
    for char_name, task_indices in char_to_tasks.items():
        tasks_processed = sum(
            1 for ti in task_indices
            if ti < len(task_results) and task_results[ti] is not None
        )

        char_folder = None
        for cf in char_folders:
            if cf.name == char_name:
                char_folder = cf
                break

        total_strokes = 0
        target_strokes = []
        if char_folder:
            stroke_info_file = char_folder / "stroke_info.json"
            if stroke_info_file.exists():
                try:
                    with open(stroke_info_file, 'r', encoding='utf-8') as f:
                        stroke_info = json.load(f)
                    total_strokes = stroke_info.get('stroke_count', 0)
                    target_candidates = [total_strokes - 1, total_strokes - 2, total_strokes - 3]
                    target_strokes = [idx for idx in target_candidates if idx >= 1]
                except:
                    pass

        char_summary[char_name] = {
            'character': char_name,
            'tasks_processed': tasks_processed,
            'total_strokes': total_strokes,
            'target_strokes': target_strokes,
            'status': 'completed' if tasks_processed > 0 else 'failed'
        }

    results_sorted = sorted(char_summary.values(), key=lambda x: x['character'])

    summary_data = {
        'total_characters': len(char_folders),
        'completed_tasks': len([r for r in task_results if r is not None]),
        'total_tasks': len(stroke_tasks),
        'processing_time_seconds': total_time,
        'results': results_sorted
    }
    safe_write_json(output_path, summary_data)

    if all_retry_stats:
        safe_write_json(retry_stats_path, all_retry_stats)
        print(f"\n📊 Retry statistics saved to: {retry_stats_path}")

    print(f"\n{'='*60}")
    print(f"✅ Done!")
    print(f"{'='*60}")
    print(f"Elapsed: {total_time:.2f} sec ({total_time/60:.1f} min)")
    print(f"Processed: {len(stroke_tasks)} tasks (character × stroke × chain length)")
    print(f"Total: {len(char_folders)} characters")
    model_name = CALL_STATS.get("model") or model
    prompt_tokens = CALL_STATS["total_prompt_tokens"]
    completion_tokens = CALL_STATS["total_completion_tokens"]
    total_tokens = prompt_tokens + completion_tokens
    print(f"Model: {model_name}")
    print(f"Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
    print(f"Summary file: {output_path}")
    print(f"Results: {output_dir_name}/ folder under each character")
    print(f"Retry stats: {retry_stats_path}")
    print(f"{'='*60}\n")

    return results_sorted

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chain-of-Frames vision chain-length experiment (last 1/2/3 strokes)"
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="../dataset",
        help="Dataset directory (default: ../dataset)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="cof_summary.json",
        help="Summary output file (default: cof_summary.json)"
    )
    parser.add_argument(
        "-w", "--max-workers",
        type=int,
        default=20,
        help="Number of parallel threads (default: 20)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume"
    )
    parser.add_argument(
        "--test",
        type=int,
        metavar="N",
        help="Test mode: process only N characters"
    )
    parser.add_argument(
        "--skip",
        type=int,
        metavar="M",
        default=0,
        help="Skip first M characters (use with --test)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name to call (default: {DEFAULT_MODEL})"
    )

    args = parser.parse_args()

    try:
        process_dataset(
            dataset_dir=args.dataset,
            output_file=args.output,
            resume=not args.no_resume,
            max_workers=args.max_workers,
            test_chars=args.test,
            skip_chars=args.skip,
            model=args.model
        )
    except KeyboardInterrupt:

        print_speed_stats()
        sys.exit(0)
