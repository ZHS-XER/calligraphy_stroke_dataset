#!/usr/bin/env python3
"""
识别 dataset 中每个字的笔画图片

纯粹收集大模型的 JSON 输出，保存到一个大文件
支持断点续传和失败重试
"""

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

# API配置
API_URL = "https://llm.q2zy.com/v1/chat/completions"
API_KEY = "MUSO_API_KEY_REMOVED"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# 全局统计数据
CALL_STATS = {
    "total_calls": 0,
    "successful_calls": 0,
    "failed_calls": 0,
    "start_time": None,
    "max_workers": 20  # 默认值，会在运行时更新
}

# 加载 prompt 模板（包含 {remaining_strokes} 占位符）
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


def image_to_base64(image_path: Path) -> str:
    """将图片转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def save_speed_test_results():
    """保存性能测试结果到 speed_test_results.json"""
    script_dir = Path(__file__).parent
    result_file = script_dir / "speed_test_results.json"
    
    # 计算统计数据
    duration = time.time() - CALL_STATS["start_time"]
    total = CALL_STATS["total_calls"]
    success = CALL_STATS["successful_calls"]
    fail = CALL_STATS["failed_calls"]
    fail_rate = fail / total if total > 0 else 0.0
    speed = success / duration if duration > 0 else 0.0
    
    # 构建新记录
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "parallelism": CALL_STATS["max_workers"],
        "total_calls": total,
        "success": success,
        "fail": fail,
        "fail_rate": round(fail_rate, 3),
        "duration_sec": round(duration, 1),
        "speed": round(speed, 2)
    }
    
    # 读取已有数据（如果文件存在）
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
    
    # 追加新记录
    data.append(record)
    
    # 写入文件
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return record, result_file


def print_speed_stats():
    """打印性能统计信息"""
    duration = time.time() - CALL_STATS["start_time"]
    total = CALL_STATS["total_calls"]
    success = CALL_STATS["successful_calls"]
    fail = CALL_STATS["failed_calls"]
    fail_rate = (fail / total * 100) if total > 0 else 0.0
    speed = success / duration if duration > 0 else 0.0
    
    print("\n" + "="*60)
    print("⚠️  检测到手动中断，正在统计当前性能...")
    print("="*60)
    print(f"📈 运行时间: {duration:.1f} 秒")
    print(f"📊 总调用: {total}")
    print(f"✅ 成功: {success}")
    print(f"❌ 失败: {fail}")
    print(f"🚫 失败率: {fail_rate:.1f}%")
    print(f"⚡ 平均速度: {speed:.2f} 次/秒")
    print("="*60)
    
    # 保存结果
    try:
        record, result_file = save_speed_test_results()
        print(f"\n✅ 当前结果已保存至 {result_file.name}\n")
    except Exception as e:
        print(f"\n❌ 保存结果失败: {str(e)}\n")


def validate_llm_output(llm_output: Dict) -> bool:
    """
    验证LLM输出是否有效
    
    Args:
        llm_output: 模型输出的字典
    
    Returns:
        是否有效
    """
    if not llm_output or not isinstance(llm_output, dict):
        return False
    
    # 检查必须字段是否存在
    if 'answer' not in llm_output or 'probabilities' not in llm_output:
        return False
    
    # 检查answer是否有效
    answer = llm_output.get('answer')
    if not answer or answer == 'null' or not isinstance(answer, str):
        return False
    
    # 检查probabilities是否有效
    probabilities = llm_output.get('probabilities')
    if not probabilities or not isinstance(probabilities, list) or len(probabilities) == 0:
        return False
    
    # 检查第一个概率项是否有效
    first_prob = probabilities[0]
    if not isinstance(first_prob, dict):
        return False
    
    if 'char' not in first_prob or not first_prob.get('char'):
        return False
    
    return True


def safe_write_json(file_path: Path, data: Dict):
    """
    安全写入JSON文件（先写临时文件，再重命名）
    
    Args:
        file_path: 目标文件路径
        data: 要写入的数据
    """
    # 创建临时文件在同一目录下
    tmp_path = file_path.with_suffix('.tmp')
    
    try:
        # 写入临时文件
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 重命名为正式文件（原子操作）
        shutil.move(str(tmp_path), str(file_path))
    except Exception as e:
        # 如果出错，清理临时文件
        if tmp_path.exists():
            tmp_path.unlink()
        raise e


def is_stroke_file_valid(file_path: Path) -> bool:
    """
    验证笔画识别结果文件是否有效
    
    Args:
        file_path: JSON文件路径
    
    Returns:
        文件是否有效（能解析且所有run的llm_output有效）
    """
    # 忽略临时文件
    if file_path.suffix == '.tmp':
        return False
    
    if not file_path.exists():
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查是否有runs字段
        if 'runs' not in data or not isinstance(data['runs'], list):
            return False
        
        # 检查每个run的llm_output是否有效
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
    """
    识别单个笔画图片一次（带增强重试机制）
    
    Args:
        image_path: 图片路径
        character: 汉字（用于日志）
        remaining_strokes: 当前笔画后还剩几笔
        max_retries: 最大重试次数（默认300次）
        initial_delay: 初始重试延迟（默认2秒）
        pbar: 进度条对象（可选）
    
    Returns:
        (完整的 API JSON 响应, 实际重试次数)，失败返回 (None, retry_count)
    """
    try:
        image_base64 = image_to_base64(image_path)
    except Exception as e:
        print(f"    ❌ 读取图片失败: {str(e)[:50]}")
        return None, 0
    
    # 动态生成 prompt，替换 {remaining_strokes} 占位符
    prompt_with_context = PROMPT_TEMPLATE.replace("{remaining_strokes}", str(remaining_strokes))
    
    data = {
        "model": "gemini-2.5-pro",
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
            response = requests.post(API_URL, headers=HEADERS, json=data, timeout=120)  # 增加超时到120秒
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                
                # 清理markdown代码块标记
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                # 解析 JSON
                try:
                    llm_output = json.loads(content)
                    
                    # ✅ 验证输出内容是否有效
                    if not validate_llm_output(llm_output):
                        print(f"    警告: 输出内容无效（answer或probabilities缺失/为空）- 重试...")
                        # 更新统计：失败
                        CALL_STATS["total_calls"] += 1
                        CALL_STATS["failed_calls"] += 1
                        if retry < max_retries - 1:
                            delay = initial_delay * (2 ** min(retry, 4))
                            time.sleep(min(delay, 30))
                            continue
                        return None, max_retries
                    
                    # 输出有效，更新进度条和统计并返回
                    CALL_STATS["total_calls"] += 1
                    CALL_STATS["successful_calls"] += 1
                    if pbar:
                        pbar.update(1)  # 成功完成一次 API 调用
                    return llm_output, retry  # 返回结果和实际重试次数
                        
                except json.JSONDecodeError as e:
                    print(f"    警告: JSON 解析失败 - {str(e)[:50]}")
                    # 更新统计：失败
                    CALL_STATS["total_calls"] += 1
                    CALL_STATS["failed_calls"] += 1
                    if retry < max_retries - 1:
                        # 指数退避，但最大等待30秒
                        delay = initial_delay * (2 ** min(retry, 4))
                        time.sleep(min(delay, 30))
                        continue
                    return None, max_retries
            else:
                # 更新统计：失败（HTTP 非200）
                CALL_STATS["total_calls"] += 1
                CALL_STATS["failed_calls"] += 1
                if retry < max_retries - 1:
                    # 指数退避，最大等待30秒
                    delay = initial_delay * (2 ** min(retry, 4))
                    wait_time = min(delay, 30)
                    time.sleep(wait_time)
                
        except requests.exceptions.Timeout:
            # 更新统计：失败（超时）
            CALL_STATS["total_calls"] += 1
            CALL_STATS["failed_calls"] += 1
            if retry < max_retries - 1:
                # 超时后等待更久
                delay = initial_delay * (2 ** min(retry, 4))
                wait_time = min(delay, 30)
                time.sleep(wait_time)
        except Exception as e:
            # 更新统计：失败（其他异常）
            CALL_STATS["total_calls"] += 1
            CALL_STATS["failed_calls"] += 1
            if retry < max_retries - 1:
                # 其他错误也增加等待时间
                delay = initial_delay * (2 ** min(retry, 4))
                wait_time = min(delay, 30)
                time.sleep(wait_time)
    
    return None, max_retries
def recognize_stroke_image_multiple(image_path: Path, character: str, 
                                    remaining_strokes: int = 0,
                                    num_runs: int = 5,
                                    pbar: Optional[tqdm] = None) -> tuple[List[Optional[Dict]], List[int]]:
    """
    对单个笔画图片运行多次识别
    
    Args:
        image_path: 图片路径
        character: 汉字（用于日志）
        remaining_strokes: 当前笔画后还剩几笔
        num_runs: 运行次数
        pbar: 进度条对象（可选）
    
    Returns:
        (识别结果列表, 重试次数列表)
    """
    results = []
    retry_counts = []
    for i in range(num_runs):
        result, retry_count = recognize_stroke_image_once(image_path, character, remaining_strokes, pbar=pbar)
        results.append(result)
        retry_counts.append(retry_count)
        if result is None:
            print(f"    ⚠️ 第 {i+1} 次识别失败")
    return results, retry_counts


def get_stroke_images(char_folder: Path) -> List[Path]:
    """
    获取字符文件夹中的笔画图片（只返回最后4个，排除整字图）
    
    Args:
        char_folder: 字符文件夹路径
    
    Returns:
        笔画图片路径列表（最后4个）
    """
    stroke_files = sorted([
        f for f in char_folder.glob("stroke_*.png")
        if f.name != "stroke_00.png"  # 排除整字图
    ])
    # 只返回最后4个笔画
    return stroke_files[-4:] if len(stroke_files) > 4 else stroke_files


def recognize_character(char_folder: Path, character: str,
                       pbar: Optional[tqdm] = None) -> Dict:
    """
    识别单个字符的笔画（只处理最后4个笔画，每张图跑5次）
    
    Args:
        char_folder: 字符文件夹路径
        character: 汉字
        pbar: 进度条对象（可选）
    
    Returns:
        包含所有笔画识别结果的字典
    """
    # print(f"\n处理字符: '{character}'")  # 注释掉，避免干扰进度条
    
    stroke_images = get_stroke_images(char_folder)
    
    if not stroke_images:
        print(f"  ⚠️ 未找到笔画图片")
        return {
            'character': character,
            'strokes_processed': 0
        }
    
    # 获取所有笔画用于统计
    all_stroke_files = sorted([
        f for f in char_folder.glob("stroke_*.png")
        if f.name != "stroke_00.png"
    ])
    
    print(f"  总笔画: {len(all_stroke_files)}，处理最后 {len(stroke_images)} 个")
    
    # 创建 recognition_1 文件夹
    recognition_dir = char_folder / "recognition_1"
    recognition_dir.mkdir(exist_ok=True)
    
    # 逐个识别（每张图跑5次）
    processed_count = 0
    total_strokes = len(all_stroke_files)
    
    # 收集重试统计数据
    retry_stats = {}
    
    for idx, stroke_file in enumerate(stroke_images):
        stroke_name = stroke_file.stem
        # 计算当前笔画的索引（从1开始）
        stroke_num = int(stroke_name.replace('stroke_', ''))
        # 计算还剩几笔
        remaining_strokes = total_strokes - stroke_num
        
        # print(f"  识别 {stroke_name} (5次, 还剩{remaining_strokes}笔)...", end=" ", flush=True)  # 注释掉
        
        # 运行5次识别，传入剩余笔画数和进度条
        results, retry_counts = recognize_stroke_image_multiple(stroke_file, character, remaining_strokes, num_runs=5, pbar=pbar)
        
        # 保存到单独的 JSON 文件
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
        
        # 使用安全写入保存到文件
        output_file = recognition_dir / f"{stroke_name}.json"
        safe_write_json(output_file, output_data)
        
        # 收集重试统计
        retry_stats[stroke_name] = [
            {'run_id': i+1, 'retry_count': rc} 
            for i, rc in enumerate(retry_counts)
        ]
        
        success_count = sum(1 for r in results if r is not None)
        # print(f"✅ ({success_count}/5)")  # 注释掉
        processed_count += 1
    
    return {
        'character': character,
        'strokes_processed': processed_count,
        'retry_stats': retry_stats
    }


def process_single_stroke(stroke_file: Path, char_folder: Path, character: str,
                         total_strokes: int, pbar: Optional[tqdm] = None) -> Dict:
    """
    处理单个笔画（笔画级并行的基本单位）
    
    Args:
        stroke_file: 笔画图片路径
        char_folder: 字符文件夹路径
        character: 汉字
        total_strokes: 该字符的总笔画数
        pbar: 进度条对象（可选）
    
    Returns:
        包含笔画识别结果的字典
    """
    stroke_name = stroke_file.stem
    stroke_num = int(stroke_name.replace('stroke_', ''))
    remaining_strokes = total_strokes - stroke_num
    
    # 创建 recognition_1 文件夹（多线程安全）
    recognition_dir = char_folder / "recognition_1"
    recognition_dir.mkdir(exist_ok=True)
    
    # 运行5次识别
    results, retry_counts = recognize_stroke_image_multiple(
        stroke_file, character, remaining_strokes, num_runs=5, pbar=pbar
    )
    
    # 构建输出数据
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
    
    # 使用安全写入保存到文件
    output_file = recognition_dir / f"{stroke_name}.json"
    safe_write_json(output_file, output_data)
    
    # 收集重试统计
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
    """加载已有的重试统计数据"""
    if retry_stats_path.exists():
        try:
            with open(retry_stats_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_retry_stats(retry_stats_path: Path, all_retry_stats: Dict):
    """保存重试统计数据"""
    with open(retry_stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_retry_stats, f, ensure_ascii=False, indent=2)


def process_dataset(dataset_dir: str = "../dataset",
                   output_file: str = "baseline_summary.json",
                   resume: bool = True,
                   max_workers: int = 20) -> List[Dict]:
    """
    处理整个数据集
    
    Args:
        dataset_dir: 数据集目录
        output_file: 输出摘要文件（记录处理进度）
        resume: 是否断点续传
        max_workers: 并行线程数
    
    Returns:
        所有字符的处理摘要列表
    """
    # 初始化全局统计
    CALL_STATS["start_time"] = time.time()
    CALL_STATS["max_workers"] = max_workers
    CALL_STATS["total_calls"] = 0
    CALL_STATS["successful_calls"] = 0
    CALL_STATS["failed_calls"] = 0
    
    script_dir = Path(__file__).parent
    dataset_path = script_dir / dataset_dir
    output_path = script_dir / output_file
    
    if not dataset_path.exists():
        print(f"❌ 错误: 数据集目录不存在 - {dataset_path}")
        return []
    
    # 获取所有汉字文件夹
    char_folders = sorted([
        d for d in dataset_path.iterdir()
        if d.is_dir() and len(d.name) > 0 and '\u4e00' <= d.name[0] <= '\u9fff'
    ])
    
    print(f"\n{'='*60}")
    print(f"🎯 收集大模型识别结果（只处理最后4个笔画，每张图5次，告知剩余笔画数）")
    print(f"{'='*60}")
    print(f"数据集: {dataset_path}")
    print(f"字符数: {len(char_folders)}")
    print(f"并行数: {max_workers}")
    print(f"输出位置: 每个字符的 recognition_1/ 文件夹")
    
    # 断点续传：检查哪些字符已经处理完成
    processed_chars = set()
    if resume:
        for char_folder in char_folders:
            recognition_dir = char_folder / "recognition_1"
            if recognition_dir.exists():
                # 检查是否所有笔画都已处理且内容有效
                stroke_images = get_stroke_images(char_folder)
                expected_files = {f"{img.stem}.json": img for img in stroke_images}
                
                # 检查所有预期文件是否存在且有效
                all_valid = True
                for json_name, img_path in expected_files.items():
                    json_path = recognition_dir / json_name
                    if not is_stroke_file_valid(json_path):
                        all_valid = False
                        break
                
                if all_valid and len(expected_files) > 0:
                    processed_chars.add(char_folder.name)
        
        if processed_chars:
            print(f"📂 已完成 {len(processed_chars)} 个字")
    
    remaining_folders = [
        d for d in char_folders
        if d.name not in processed_chars
    ]
    
    print(f"待处理: {len(remaining_folders)}")
    print(f"{'='*60}\n")
    
    if len(remaining_folders) == 0:
        print("✅ 所有字符都已处理完成！")
        return []
    
    # 计算总的 API 调用次数（每个字符最多4个笔画，每个笔画5次）
    total_api_calls = 0
    for char_folder in remaining_folders:
        stroke_images = get_stroke_images(char_folder)
        total_api_calls += len(stroke_images) * 5  # 每个笔画5次
    
    print(f"预计 API 调用: {total_api_calls} 次")
    print(f"{'='*60}\n")
    
    # 确定重试统计文件路径
    if output_file:
        retry_stats_path = script_dir / output_file.replace('.json', '_retry.json')
    else:
        retry_stats_path = script_dir / "retry_statistics.json"
    
    # 加载已有的重试统计（如果是 resume 模式）
    all_retry_stats = load_retry_stats(retry_stats_path) if resume else {}
    
    results = []
    start_time = time.time()
    
    # ✅ 笔画级并行：收集所有待处理的笔画任务
    stroke_tasks = []
    char_to_strokes = {}  # 记录每个字符对应的笔画任务索引
    
    for char_folder in remaining_folders:
        stroke_images = get_stroke_images(char_folder)
        if not stroke_images:
            continue
        
        # 获取总笔画数
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
    
    print(f"待处理笔画任务: {len(stroke_tasks)} 个")
    print(f"{'='*60}\n")
    
    # 并行处理所有笔画（笔画级并行）
    stroke_results = [None] * len(stroke_tasks)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建进度条，以 API 调用次数为单位（每个笔画5次）
        total_api_calls = len(stroke_tasks) * 5
        with tqdm(total=total_api_calls, desc="API 调用", unit="次", 
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
                    print(f"\n❌ 笔画处理失败: {str(e)[:100]}")
                    stroke_results[idx] = None
    
    # 汇总结果：按字符组织
    all_retry_stats = load_retry_stats(retry_stats_path) if resume else {}
    
    for char_name, task_indices in char_to_strokes.items():
        char_retry_stats = {}
        strokes_processed = 0
        
        for task_idx in task_indices:
            stroke_result = stroke_results[task_idx]
            if stroke_result:
                strokes_processed += 1
                # 合并重试统计
                if 'retry_stats' in stroke_result:
                    char_retry_stats.update(stroke_result['retry_stats'])
        
        # 保存该字符的重试统计
        if char_retry_stats:
            all_retry_stats[char_name] = char_retry_stats
            save_retry_stats(retry_stats_path, all_retry_stats)
        
        # 添加到结果列表
        results.append({
            'character': char_name,
            'strokes_processed': strokes_processed
        })
    
    total_time = time.time() - start_time
    completed_count = len(processed_chars) + len(results)
    
    # 最终保存摘要（使用安全写入）
    results_sorted = sorted(results, key=lambda x: x['character'])
    summary_data = {
        'total_characters': len(char_folders),
        'processed_characters': completed_count,
        'processing_time_seconds': total_time,
        'results': results_sorted
    }
    safe_write_json(output_path, summary_data)
    
    # 最终确认保存重试统计（理论上已经保存过了，这里再确认一次）
    if all_retry_stats:
        save_retry_stats(retry_stats_path, all_retry_stats)
        print(f"\n📊 重试统计已保存到: {retry_stats_path}")
    
    print(f"\n{'='*60}")
    print(f"✅ 完成！")
    print(f"{'='*60}")
    print(f"耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
    print(f"处理: {len(remaining_folders)} 个字")
    print(f"总计: {completed_count}/{len(char_folders)}")
    print(f"摘要文件: {output_path}")
    print(f"识别结果: 各字符的 recognition_1/ 文件夹")
    print(f"重试统计: {retry_stats_path}")
    print(f"{'='*60}\n")
    
    return results_sorted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="收集大模型对笔画图片的识别结果"
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="../dataset",
        help="数据集目录 (默认: ../dataset)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="baseline_summary.json",
        help="输出摘要文件 (默认: baseline_summary.json)"
    )
    parser.add_argument(
        "-w", "--max-workers",
        type=int,
        default=20,
        help="并行线程数 (默认: 20)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="不使用断点续传"
    )
    parser.add_argument(
        "--test",
        type=int,
        metavar="N",
        help="测试模式：只处理 N 个字符"
    )
    parser.add_argument(
        "--skip",
        type=int,
        metavar="M",
        default=0,
        help="跳过前 M 个字符（配合 --test 使用）"
    )
    
    args = parser.parse_args()
    
    # 用 try-except 包裹主逻辑以捕获中断
    try:
        if args.test:
            skip = args.skip if args.skip else 0
            print(f"\n⚠️  测试模式：跳过前 {skip} 个字符，处理 {args.test} 个字符 (字符 {skip+1}-{skip+args.test})\n")
            
            # 初始化全局统计
            CALL_STATS["start_time"] = time.time()
            CALL_STATS["max_workers"] = args.max_workers
            CALL_STATS["total_calls"] = 0
            CALL_STATS["successful_calls"] = 0
            CALL_STATS["failed_calls"] = 0
            
            script_dir = Path(__file__).parent
            dataset_path = script_dir / args.dataset
            
            char_folders = sorted([
                d for d in dataset_path.iterdir()
                if d.is_dir() and len(d.name) > 0 and '\u4e00' <= d.name[0] <= '\u9fff'
            ])[skip:skip+args.test]
            
            # 确定重试统计文件路径
            if args.output:
                retry_stats_path = script_dir / args.output.replace('.json', '_retry.json')
            else:
                retry_stats_path = script_dir / "test_retry_statistics.json"
            
            # 加载已有的重试统计
            all_retry_stats = load_retry_stats(retry_stats_path)
            
            results = []
            test_output = script_dir / f"test_{args.test}.json"
            
            for char_folder in char_folders:
                # 获取笔画
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
                
                # 处理每个笔画
                for stroke_file in stroke_images:
                    stroke_result = process_single_stroke(
                        stroke_file, char_folder, char_folder.name, total_strokes
                    )
                    if stroke_result:
                        strokes_processed += 1
                        if 'retry_stats' in stroke_result:
                            char_retry_stats.update(stroke_result['retry_stats'])
                
                # 保存重试统计
                if char_retry_stats:
                    all_retry_stats[char_folder.name] = char_retry_stats
                    save_retry_stats(retry_stats_path, all_retry_stats)
                
                results.append({
                    'character': char_folder.name,
                    'strokes_processed': strokes_processed
                })
                
                # 每处理完一个字符就保存测试结果（使用安全写入）
                safe_write_json(test_output, results)
            
            print(f"\n✅ 测试结果: {test_output}")
            print(f"📊 重试统计: {retry_stats_path}")
        else:
            process_dataset(
                dataset_dir=args.dataset,
                output_file=args.output,
                resume=not args.no_resume,
                max_workers=args.max_workers
            )
    except KeyboardInterrupt:
        # 捕获 Ctrl+C 中断
        print_speed_stats()
        sys.exit(0)
