import requests
import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm
import os
import tempfile
import shutil
import sys
from datetime import datetime
import re

API_URL = os.environ.get("MUSO_API_URL", "")
API_KEY = os.environ.get("MUSO_API_KEY", "")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

DEFAULT_MODEL = "gemini-2.5-pro"
CURRENT_MODEL = DEFAULT_MODEL
MODEL_DIR_ALIASES = {
    "gpt-5-2025-08-07": "gpt-5",
}

def get_output_dir_name(model: str) -> str:

    model = MODEL_DIR_ALIASES.get(model, model or "")
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", model)
    if model == DEFAULT_MODEL:
        return "recognition_1"
    return f"recognition_{normalized or 'default'}"

CURRENT_OUTPUT_DIR_NAME = get_output_dir_name(CURRENT_MODEL)
RUNS_PER_STROKE = 1
RETRY_DELAYS = [2, 3, 4, 60]

CALL_STATS = {
    "total_calls": 0,
    "successful_calls": 0,
    "failed_calls": 0,
    "start_time": None,
    "max_workers": 20,
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_tokens": 0
}

def get_retry_delay(attempt: int) -> float:

    if attempt < len(RETRY_DELAYS):
        return RETRY_DELAYS[attempt]
    return RETRY_DELAYS[-1]

PROMPT_TEMPLATE = """任务：识别单帧书法图像中的汉字（该帧信息已足够唯一确定该字）。

上下文：
- 本图像是汉字书法教学视频的截图。
- 当前帧包含足以判断目标汉字的笔画与结构。
- 此时还剩{remaining_strokes}个笔划没有写完。

输出规范（务必严格遵守）：
- 只输出一个 JSON 对象，且不包含任何多余文字、换行前缀或注释。
- 字段与取值：
  "answer": "<单个汉字>",                       // 最终判定（必须与概率列表首项一致）
  "probabilities": [                            // 按概率降序，长度 3
    {{"char":"<汉字1>","p":0.xx}},
    {{"char":"<汉字2>","p":0.xx}},
    {{"char":"<汉字3>","p":0.xx}}
  ],
  "rationale": "<推断依据>"

严格格式要求：
- 概率 p 为 0–1 之间的小数，保留两位；所有 p 之和必须等于 1.00（四舍五入后）。
- 若 "answer" ≠ probabilities[0].char，则令 "answer" = probabilities[0].char。
- 若可能的汉字数 < 3，用 null 代替汉字，概率则用 0.00 填充。
- 仅可输出中文汉字作为候选，不得输出词语、拼音或符号。

校准指引（内部执行，最终仅输出 JSON）：
- 先为 Top-K 候选分配相对打分 s_i ∈ {{1..9}}，再归一化为概率 p_i = s_i / Σ s_i。
- 若最高概率 < 0.50，仍给出最可能答案
- 在 "rationale" 中简述推断依据以及歧义来源。

现在输出符合上述规范的 JSON。"""

def extract_json_object(text: str) -> Optional[str]:

    if not text:
        return None
    in_string = False
    escape = False
    depth = 0
    start_idx: Optional[int] = None
    for idx, ch in enumerate(text):
        if ch == "\\" and not escape:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
        escape = False
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start_idx = idx
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start_idx is not None:
                candidate = text[start_idx : idx + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue
    return None

def image_to_base64(image_path: Path) -> str:

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def save_speed_test_results():

    script_dir = Path(__file__).parent
    result_file = script_dir / "speed_test_results.json"

    duration = time.time() - CALL_STATS["start_time"]
    total = CALL_STATS["total_calls"]
    success = CALL_STATS["successful_calls"]
    fail = CALL_STATS["failed_calls"]
    fail_rate = fail / total if total > 0 else 0.0
    speed = success / duration if duration > 0 else 0.0
    input_tokens = CALL_STATS["total_input_tokens"]
    output_tokens = CALL_STATS["total_output_tokens"]
    total_tokens = CALL_STATS["total_tokens"]
    tokens_per_sec = total_tokens / duration if duration > 0 else 0.0

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "parallelism": CALL_STATS["max_workers"],
        "total_calls": total,
        "success": success,
        "fail": fail,
        "fail_rate": round(fail_rate, 3),
        "duration_sec": round(duration, 1),
        "speed": round(speed, 2),
        "model": CURRENT_MODEL,
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": round(tokens_per_sec, 2)
    }

    if result_file.exists():
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
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
    input_tokens = CALL_STATS["total_input_tokens"]
    output_tokens = CALL_STATS["total_output_tokens"]
    total_tokens = CALL_STATS["total_tokens"]
    tokens_per_sec = total_tokens / duration if duration > 0 else 0.0

    print("\n" + "="*60)
    print("⚠️  Detected manual interruption; summarizing performance...")
    print("="*60)
    print(f"📈 Runtime: {duration:.1f} sec")
    print(f"📊 Total calls: {total}")
    print(f"✅ Success: {success}")
    print(f"❌ Failed: {fail}")
    print(f"🚫 Failure rate: {fail_rate:.1f}%")
    print(f"⚡ Avg speed: {speed:.2f} calls/sec")
    print(f"🧮 Prompt tokens: {input_tokens}")
    print(f"📝 Completion tokens: {output_tokens}")
    print(f"🧱 Total tokens: {total_tokens} ({tokens_per_sec:.2f}/sec)")
    print("="*60)

    try:
        record, result_file = save_speed_test_results()
        print(f"\n✅ Current results saved to {result_file.name}\n")
    except Exception as e:
        print(f"\n❌ Failed to save results: {str(e)}\n")

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

def is_stroke_file_valid(file_path: Path) -> bool:

    if file_path.suffix == '.tmp':
        return False

    if not file_path.exists():
        return False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'runs' not in data or not isinstance(data['runs'], list):
            return False

        for run in data['runs']:
            if 'llm_output' not in run:
                return False

            llm_output = run['llm_output']
            if not validate_llm_output(llm_output):
                return False

        return True
    except Exception:
        return False

def recognize_stroke_image_once(image_path: Path, character: str,
                                remaining_strokes: int = 0,
                                max_retries: int = 300, initial_delay: float = 2.0,
                                pbar: Optional[tqdm] = None) -> tuple[Optional[Dict], int]:

    try:
        image_base64 = image_to_base64(image_path)
    except Exception as e:
        print(f"    ❌ Failed to read image: {str(e)[:50]}")
        return None, 0

    prompt_with_context = PROMPT_TEMPLATE.replace("{remaining_strokes}", str(remaining_strokes))

    data = {
        "model": CURRENT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是一个汉字识别助手"
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_with_context},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
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
                usage = result.get("usage") if isinstance(result, dict) else None
                if isinstance(usage, dict):
                    def _to_int(value):
                        try:
                            return int(value)
                        except (TypeError, ValueError):
                            return 0
                    prompt_tokens_raw = usage.get("prompt_tokens", usage.get("input_tokens"))
                    completion_tokens_raw = usage.get("completion_tokens", usage.get("output_tokens"))
                    total_tokens_raw = usage.get("total_tokens")

                    prompt_tokens = _to_int(prompt_tokens_raw)
                    completion_tokens = _to_int(completion_tokens_raw)
                    if total_tokens_raw is None:
                        total_tokens = prompt_tokens + completion_tokens
                    else:
                        total_tokens = _to_int(total_tokens_raw)
                        if total_tokens == 0 and (prompt_tokens or completion_tokens):
                            total_tokens = prompt_tokens + completion_tokens

                    CALL_STATS["total_input_tokens"] += prompt_tokens
                    CALL_STATS["total_output_tokens"] += completion_tokens
                    CALL_STATS["total_tokens"] += total_tokens
                content = result["choices"][0]["message"]["content"].strip()

                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                llm_output = None
                try:
                    llm_output = json.loads(content)
                except json.JSONDecodeError:
                    extracted = extract_json_object(content)
                    if extracted:
                        try:
                            llm_output = json.loads(extracted)
                        except json.JSONDecodeError:
                            llm_output = None
                    if llm_output is None:
                        snippet = content.strip().replace("\n", " ")
                        snippet = snippet[:160] + ("..." if len(snippet) > 160 else "")
                        print(
                            f"    Warning: JSON parse failed - could not parse model output snippet: {snippet or '<empty response>'}"
                        )

                        CALL_STATS["total_calls"] += 1
                        CALL_STATS["failed_calls"] += 1
                        if retry < max_retries - 1:

                            wait_time = get_retry_delay(retry)
                            time.sleep(wait_time)
                            continue
                        return None, max_retries

                if not validate_llm_output(llm_output):
                    print("    Warning: output invalid (answer/probabilities missing or empty) - retrying...")

                    CALL_STATS["total_calls"] += 1
                    CALL_STATS["failed_calls"] += 1
                    if retry < max_retries - 1:
                        wait_time = get_retry_delay(retry)
                        time.sleep(wait_time)
                        continue
                    return None, max_retries

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
def recognize_stroke_image_multiple(image_path: Path, character: str,
                                    remaining_strokes: int = 0,
                                    num_runs: int = RUNS_PER_STROKE,
                                    pbar: Optional[tqdm] = None) -> tuple[List[Optional[Dict]], List[int]]:

    results = []
    retry_counts = []
    for i in range(num_runs):
        result, retry_count = recognize_stroke_image_once(image_path, character, remaining_strokes, pbar=pbar)
        results.append(result)
        retry_counts.append(retry_count)
        if result is None:
            print(f"    ⚠️ Attempt {i+1} failed")
    return results, retry_counts

def get_stroke_images(char_folder: Path) -> List[Path]:

    stroke_files = sorted([
        f for f in char_folder.glob("stroke_*.png")
        if f.name != "stroke_00.png"
    ])

    return stroke_files[-4:] if len(stroke_files) > 4 else stroke_files

def recognize_character(char_folder: Path, character: str,
                       pbar: Optional[tqdm] = None) -> Dict:

    stroke_images = get_stroke_images(char_folder)

    if not stroke_images:
        print("  ⚠️ No stroke images found")
        return {
            'character': character,
            'strokes_processed': 0
        }

    all_stroke_files = sorted([
        f for f in char_folder.glob("stroke_*.png")
        if f.name != "stroke_00.png"
    ])

    print(f"  Total strokes: {len(all_stroke_files)}; processing the final {len(stroke_images)} images")

    recognition_dir = char_folder / CURRENT_OUTPUT_DIR_NAME
    recognition_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    total_strokes = len(all_stroke_files)

    retry_stats = {}

    for idx, stroke_file in enumerate(stroke_images):
        stroke_name = stroke_file.stem

        stroke_num = int(stroke_name.replace('stroke_', ''))

        remaining_strokes = total_strokes - stroke_num

        results, retry_counts = recognize_stroke_image_multiple(
            stroke_file, character, remaining_strokes, pbar=pbar
        )

        output_data = {
            'character': character,
            'stroke_name': stroke_name,
            'image_path': str(stroke_file.relative_to(char_folder.parent.parent)),
            'remaining_strokes': remaining_strokes,
            'runs': []
        }

        for i, llm_output in enumerate(results, 1):
            output_data['runs'].append({
                'run_id': i,
                'llm_output': llm_output
            })

        output_file = recognition_dir / f"{stroke_name}.json"
        safe_write_json(output_file, output_data)

        retry_stats[stroke_name] = [
            {'run_id': i+1, 'retry_count': rc}
            for i, rc in enumerate(retry_counts)
        ]

        success_count = sum(1 for r in results if r is not None)

        processed_count += 1

    return {
        'character': character,
        'strokes_processed': processed_count,
        'retry_stats': retry_stats
    }

def process_single_stroke(stroke_file: Path, char_folder: Path, character: str,
                         total_strokes: int, pbar: Optional[tqdm] = None) -> Dict:

    stroke_name = stroke_file.stem
    stroke_num = int(stroke_name.replace('stroke_', ''))
    remaining_strokes = total_strokes - stroke_num

    recognition_dir = char_folder / CURRENT_OUTPUT_DIR_NAME
    recognition_dir.mkdir(parents=True, exist_ok=True)

    results, retry_counts = recognize_stroke_image_multiple(
        stroke_file, character, remaining_strokes, pbar=pbar
    )

    output_data = {
        'character': character,
        'stroke_name': stroke_name,
        'image_path': str(stroke_file.relative_to(char_folder.parent.parent)),
        'remaining_strokes': remaining_strokes,
        'runs': []
    }

    for i, llm_output in enumerate(results, 1):
        output_data['runs'].append({
            'run_id': i,
            'llm_output': llm_output
        })

    output_file = recognition_dir / f"{stroke_name}.json"
    safe_write_json(output_file, output_data)

    retry_stats = {
        stroke_name: [
            {'run_id': i+1, 'retry_count': rc}
            for i, rc in enumerate(retry_counts)
        ]
    }

    success_count = sum(1 for r in results if r is not None)

    return {
        'character': character,
        'stroke_name': stroke_name,
        'success_count': success_count,
        'retry_stats': retry_stats
    }

def load_retry_stats(retry_stats_path: Path) -> Dict:

    if retry_stats_path.exists():
        try:
            with open(retry_stats_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_retry_stats(retry_stats_path: Path, all_retry_stats: Dict):

    with open(retry_stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_retry_stats, f, ensure_ascii=False, indent=2)

def process_dataset(dataset_dir: str = "../dataset",
                   output_file: str = "baseline_summary.json",
                   resume: bool = True,
                   max_workers: int = 20) -> List[Dict]:

    CALL_STATS["start_time"] = time.time()
    CALL_STATS["max_workers"] = max_workers
    CALL_STATS["total_calls"] = 0
    CALL_STATS["successful_calls"] = 0
    CALL_STATS["failed_calls"] = 0
    CALL_STATS["total_input_tokens"] = 0
    CALL_STATS["total_output_tokens"] = 0
    CALL_STATS["total_tokens"] = 0

    script_dir = Path(__file__).parent
    dataset_path = script_dir / dataset_dir
    output_path = script_dir / output_file

    if not dataset_path.exists():
        print(f"❌ Error: dataset directory does not exist - {dataset_path}")
        return []

    char_folders = sorted([
        d for d in dataset_path.iterdir()
        if d.is_dir() and len(d.name) > 0 and '\u4e00' <= d.name[0] <= '\u9fff'
    ])

    print(f"\n{'='*60}")
    print(f"🎯 Collecting LLM recognition results (last 4 strokes, {RUNS_PER_STROKE} run(s) per image, informing remaining strokes)")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Characters: {len(char_folders)}")
    print(f"Parallelism: {max_workers}")
    print(f"Output location: {CURRENT_OUTPUT_DIR_NAME}/ under each character")

    processed_chars = set()
    if resume:
        for char_folder in char_folders:
            recognition_dir = char_folder / CURRENT_OUTPUT_DIR_NAME
            if recognition_dir.exists():

                stroke_images = get_stroke_images(char_folder)
                expected_files = {f"{img.stem}.json": img for img in stroke_images}

                all_valid = True
                for json_name, img_path in expected_files.items():
                    json_path = recognition_dir / json_name
                    if not is_stroke_file_valid(json_path):
                        all_valid = False
                        break

                if all_valid and len(expected_files) > 0:
                    processed_chars.add(char_folder.name)

        if processed_chars:
            print(f"📂 Completed {len(processed_chars)} characters")

    remaining_folders = [
        d for d in char_folders
        if d.name not in processed_chars
    ]

    print(f"Remaining: {len(remaining_folders)}")
    print(f"{'='*60}\n")

    if len(remaining_folders) == 0:
        print("✅ All characters have been processed!")
        return []

    total_api_calls = 0
    for char_folder in remaining_folders:
        stroke_images = get_stroke_images(char_folder)
        total_api_calls += len(stroke_images) * RUNS_PER_STROKE

    print(f"Estimated API calls: {total_api_calls}")
    print(f"{'='*60}\n")

    if output_file:
        retry_stats_path = script_dir / output_file.replace('.json', '_retry.json')
    else:
        retry_stats_path = script_dir / "retry_statistics.json"

    all_retry_stats = load_retry_stats(retry_stats_path) if resume else {}

    results = []
    start_time = time.time()

    stroke_tasks = []
    char_to_strokes = {}

    for char_folder in remaining_folders:
        stroke_images = get_stroke_images(char_folder)
        if not stroke_images:
            continue

        all_stroke_files = sorted([
            f for f in char_folder.glob("stroke_*.png")
            if f.name != "stroke_00.png"
        ])
        total_strokes = len(all_stroke_files)

        char_name = char_folder.name
        char_to_strokes[char_name] = []

        for stroke_file in stroke_images:
            task_idx = len(stroke_tasks)
            stroke_tasks.append({
                'stroke_file': stroke_file,
                'char_folder': char_folder,
                'character': char_name,
                'total_strokes': total_strokes
            })
            char_to_strokes[char_name].append(task_idx)

    print(f"Stroke tasks to process: {len(stroke_tasks)}")
    print(f"{'='*60}\n")

    stroke_results = [None] * len(stroke_tasks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        total_api_calls = len(stroke_tasks) * RUNS_PER_STROKE
        with tqdm(total=total_api_calls, desc="API calls", unit="calls",
                  ncols=100, smoothing=0.1) as pbar:
            future_to_idx = {
                executor.submit(
                    process_single_stroke,
                    task['stroke_file'],
                    task['char_folder'],
                    task['character'],
                    task['total_strokes'],
                    pbar
                ): idx
                for idx, task in enumerate(stroke_tasks)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    stroke_results[idx] = result
                except Exception as e:
                    print(f"\n❌ Stroke processing failed: {str(e)[:100]}")
                    stroke_results[idx] = None

    all_retry_stats = load_retry_stats(retry_stats_path) if resume else {}

    for char_name, task_indices in char_to_strokes.items():
        char_retry_stats = {}
        strokes_processed = 0

        for task_idx in task_indices:
            stroke_result = stroke_results[task_idx]
            if stroke_result:
                strokes_processed += 1

                if 'retry_stats' in stroke_result:
                    char_retry_stats.update(stroke_result['retry_stats'])

        if char_retry_stats:
            all_retry_stats[char_name] = char_retry_stats
            save_retry_stats(retry_stats_path, all_retry_stats)

        results.append({
            'character': char_name,
            'strokes_processed': strokes_processed
        })

    total_time = time.time() - start_time
    completed_count = len(processed_chars) + len(results)

    results_sorted = sorted(results, key=lambda x: x['character'])
    summary_data = {
        'total_characters': len(char_folders),
        'processed_characters': completed_count,
        'processing_time_seconds': total_time,
        'results': results_sorted
    }
    safe_write_json(output_path, summary_data)

    if all_retry_stats:
        save_retry_stats(retry_stats_path, all_retry_stats)
        print(f"\n📊 Retry statistics saved to: {retry_stats_path}")

    print(f"\n{'='*60}")
    print("✅ Done!")
    print(f"{'='*60}")
    print(f"Elapsed: {total_time:.2f} sec ({total_time/60:.1f} min)")
    print(f"Processed: {len(remaining_folders)} characters")
    print(f"Total: {completed_count}/{len(char_folders)}")
    print(f"Summary file: {output_path}")
    print(f"Results: {CURRENT_OUTPUT_DIR_NAME}/ folder under each character")
    print(f"Retry stats: {retry_stats_path}")
    print(f"{'='*60}\n")

    try:
        record, result_file = save_speed_test_results()
        print(f"📊 Performance record saved to {result_file}")
    except Exception as e:
        print(f"⚠️ Failed to write performance record: {e}")

    return results_sorted

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect LLM recognition outputs on stroke images"
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
        default="baseline_summary.json",
        help="Summary output file (default: baseline_summary.json)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name to call (default: {DEFAULT_MODEL})"
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

    args = parser.parse_args()
    CURRENT_MODEL = args.model
    CURRENT_OUTPUT_DIR_NAME = get_output_dir_name(CURRENT_MODEL)

    print(f"📊 Current model: {CURRENT_MODEL}")
    print(f"📂 Output directory: {CURRENT_OUTPUT_DIR_NAME}/")

    try:
        if args.test:
            skip = args.skip if args.skip else 0
            print(f"\n⚠️  Test mode: skipping first {skip} characters, processing {args.test} characters (indices {skip+1}-{skip+args.test})\n")

            CALL_STATS["start_time"] = time.time()
            CALL_STATS["max_workers"] = args.max_workers
            CALL_STATS["total_calls"] = 0
            CALL_STATS["successful_calls"] = 0
            CALL_STATS["failed_calls"] = 0
            CALL_STATS["total_input_tokens"] = 0
            CALL_STATS["total_output_tokens"] = 0
            CALL_STATS["total_tokens"] = 0

            script_dir = Path(__file__).parent
            dataset_path = script_dir / args.dataset

            char_folders = sorted([
                d for d in dataset_path.iterdir()
                if d.is_dir() and len(d.name) > 0 and '\u4e00' <= d.name[0] <= '\u9fff'
            ])[skip:skip+args.test]

            if args.output:
                retry_stats_path = script_dir / args.output.replace('.json', '_retry.json')
            else:
                retry_stats_path = script_dir / "test_retry_statistics.json"

            all_retry_stats = load_retry_stats(retry_stats_path)

            results = []
            test_output = script_dir / f"test_{args.test}.json"

            for char_folder in char_folders:

                stroke_images = get_stroke_images(char_folder)
                if not stroke_images:
                    continue

                all_stroke_files = sorted([
                    f for f in char_folder.glob("stroke_*.png")
                    if f.name != "stroke_00.png"
                ])
                total_strokes = len(all_stroke_files)

                char_retry_stats = {}
                strokes_processed = 0

                for stroke_file in stroke_images:
                    stroke_result = process_single_stroke(
                        stroke_file, char_folder, char_folder.name, total_strokes
                    )
                    if stroke_result:
                        strokes_processed += 1
                        if 'retry_stats' in stroke_result:
                            char_retry_stats.update(stroke_result['retry_stats'])

                if char_retry_stats:
                    all_retry_stats[char_folder.name] = char_retry_stats
                    save_retry_stats(retry_stats_path, all_retry_stats)

                results.append({
                    'character': char_folder.name,
                    'strokes_processed': strokes_processed
                })

                safe_write_json(test_output, results)

            print(f"\n✅ Test results: {test_output}")
            print(f"📊 Retry stats: {retry_stats_path}")
            try:
                record, result_file = save_speed_test_results()
                print(f"📊 Performance record written to {result_file}")
            except Exception as e:
                print(f"⚠️ Failed to write performance record: {e}")
        else:
            process_dataset(
                dataset_dir=args.dataset,
                output_file=args.output,
                resume=not args.no_resume,
                max_workers=args.max_workers
            )
    except KeyboardInterrupt:

        print_speed_stats()
        sys.exit(0)
