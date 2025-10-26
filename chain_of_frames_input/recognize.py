#!/usr/bin/env python3
"""
Chain-of-Frames 输入识别脚本（增强版）

处理每个字符的倒数第1、2、3笔，每笔均匀采样5帧作为输入
同步 baseline_1 的稳定性改进：重试机制、安全写入、断点续传、性能监控
结果保存到各字符的 COF_input/ 文件夹
"""

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
from tqdm import tqdm
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

# 加载 prompt 模板
def load_prompt_template() -> str:
    """加载当前目录下的 prompt.md 文件"""
    prompt_file = Path(__file__).parent / "prompt.md"
    if not prompt_file.exists():
        raise FileNotFoundError(f"找不到 prompt 文件: {prompt_file}")
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()

PROMPT_TEMPLATE = load_prompt_template()


def image_to_base64(image_path: Path) -> str:
    """将图片转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


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


def is_stroke_file_valid(file_path: Path, expected_runs: int = 5) -> bool:
    """
    验证笔画识别结果文件是否有效
    
    Args:
        file_path: JSON文件路径
        expected_runs: 期望的运行次数
    
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
        
        # 检查运行次数是否足够
        if len(data['runs']) < expected_runs:
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


def select_frame_images(char_folder: Path, target_stroke: int, total_strokes: int) -> List[Path]:
    """
    选择输入帧图片（新逻辑：基于目标笔画）
    
    从 stroke_01 到 stroke_{target_stroke}，均匀采样5张图片
    
    Args:
        char_folder: 字符文件夹路径
        target_stroke: 目标笔画索引（从1开始，如倒数第1笔对应 total_strokes）
        total_strokes: 该字符的总笔画数
    
    Returns:
        选中的图片路径列表（按笔画顺序，共5张）
    """
    # 候选范围: stroke_01 到 stroke_{target_stroke}
    candidate_range = list(range(1, target_stroke + 1))  # [1, 2, ..., target_stroke]
    
    if len(candidate_range) == 0:
        return []
    
    # 选择策略：均匀采样5张
    if len(candidate_range) <= 5:
        # 不足5张，全部选择
        selected_indices = candidate_range
    else:
        # 均匀采样5张
        step = (len(candidate_range) - 1) / 4  # 分成5段
        selected_indices = [
            candidate_range[0],  # 第一张
            candidate_range[round(step)],
            candidate_range[round(2 * step)],
            candidate_range[round(3 * step)],
            candidate_range[-1]  # 最后一张（目标笔画）
        ]
        # 去重（以防 round 导致重复）
        selected_indices = sorted(list(set(selected_indices)))
    
    # 构建图片路径列表
    selected_images = []
    for idx in selected_indices:
        img_path = char_folder / f"stroke_{idx:02d}.png"
        if img_path.exists():
            selected_images.append(img_path)
    
    return selected_images


def recognize_with_frames(images: List[Path], character: str, remaining_strokes: int,
                         max_retries: int = 300, initial_delay: float = 2.0,
                         pbar: Optional[tqdm] = None) -> tuple[Optional[Dict], int]:
    """
    使用多帧图片进行识别（带增强重试机制）
    
    Args:
        images: 图片路径列表（按笔画顺序）
        character: 汉字（用于日志）
        remaining_strokes: 目标笔画之后还剩几个笔画
        max_retries: 最大重试次数（默认300次）
        initial_delay: 初始重试延迟（默认2秒）
        pbar: 进度条对象（可选）
    
    Returns:
        (完整的 API JSON 响应, 实际重试次数)，失败返回 (None, retry_count)
    """
    try:
        # 转换所有图片为 base64
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
        print(f"    ❌ 读取图片失败: {str(e)[:50]}")
        return None, 0
    
    # 替换 prompt 中的 {x} 为 remaining_strokes
    prompt_with_context = PROMPT_TEMPLATE.replace("{x}", str(remaining_strokes))
    
    # 构建消息内容：text + 多张图片
    user_content = [{"type": "text", "text": prompt_with_context}] + image_contents
    
    data = {
        "model": "gemini-2.5-pro",
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
                    
                    # 验证输出有效性
                    if validate_llm_output(llm_output):
                        # 更新统计：成功
                        CALL_STATS["total_calls"] += 1
                        CALL_STATS["successful_calls"] += 1
                        if pbar:
                            pbar.update(1)
                        return llm_output, retry
                    else:
                        # 输出无效，重试
                        CALL_STATS["total_calls"] += 1
                        CALL_STATS["failed_calls"] += 1
                        if retry < max_retries - 1:
                            delay = initial_delay * (2 ** min(retry, 4))
                            time.sleep(min(delay, 30))
                            continue
                        return None, retry + 1
                        
                except json.JSONDecodeError as e:
                    # 更新统计：失败（JSON解析错误）
                    CALL_STATS["total_calls"] += 1
                    CALL_STATS["failed_calls"] += 1
                    if retry < max_retries - 1:
                        delay = initial_delay * (2 ** min(retry, 4))
                        time.sleep(min(delay, 30))
                        continue
                    return None, retry + 1
            else:
                # 更新统计：失败（HTTP错误）
                CALL_STATS["total_calls"] += 1
                CALL_STATS["failed_calls"] += 1
                if retry < max_retries - 1:
                    delay = initial_delay * (2 ** min(retry, 4))
                    wait_time = min(delay, 30)
                    time.sleep(wait_time)
                
        except requests.exceptions.Timeout:
            # 更新统计：失败（超时）
            CALL_STATS["total_calls"] += 1
            CALL_STATS["failed_calls"] += 1
            if retry < max_retries - 1:
                delay = initial_delay * (2 ** min(retry, 4))
                wait_time = min(delay, 30)
                time.sleep(wait_time)
        except Exception as e:
            # 更新统计：失败（其他异常）
            CALL_STATS["total_calls"] += 1
            CALL_STATS["failed_calls"] += 1
            if retry < max_retries - 1:
                delay = initial_delay * (2 ** min(retry, 4))
                wait_time = min(delay, 30)
                time.sleep(wait_time)
    
    return None, max_retries


def recognize_single_stroke(char_folder: Path, character: str,
                          target_stroke: int, total_strokes: int,
                          num_runs: int = 5,
                          pbar: Optional[tqdm] = None) -> Dict:
    """
    识别单个笔画（笔画级并行的基本单位）
    
    Args:
        char_folder: 字符文件夹路径
        character: 汉字
        target_stroke: 目标笔画索引（从1开始）
        total_strokes: 该字符的总笔画数
        num_runs: 运行次数（默认5次）
        pbar: 进度条对象（可选）
    
    Returns:
        包含笔画识别结果的字典
    """
    stroke_name = f"stroke_{target_stroke:02d}"
    
    # 计算还剩几笔
    remaining_strokes = total_strokes - target_stroke
    
    # 创建 COF_input 文件夹（多线程安全）
    output_dir = char_folder / "COF_input"
    output_dir.mkdir(exist_ok=True)
    
    # 选择输入图片
    try:
        input_images = select_frame_images(char_folder, target_stroke, total_strokes)
    except Exception as e:
        print(f"  ⚠️ {character} {stroke_name}: 选择图片失败 - {str(e)}")
        return {
            'character': character,
            'stroke_name': stroke_name,
            'status': 'image_selection_failed',
            'error': str(e)
        }
    
    if not input_images:
        print(f"  ⚠️ {character} {stroke_name}: 没有可用的图片")
        return {
            'character': character,
            'stroke_name': stroke_name,
            'status': 'no_images'
        }
    
    # 运行多次识别
    results = []
    retry_counts = []
    for i in range(num_runs):
        result, retry_count = recognize_with_frames(
            input_images, character, remaining_strokes, pbar=pbar
        )
        results.append(result)
        retry_counts.append(retry_count)
        if result is None:
            pass  # 静默失败，避免干扰进度条
    
    # 构建输出数据
    output_data = {
        'character': character,
        'stroke_name': stroke_name,
        'target_stroke': target_stroke,
        'remaining_strokes': remaining_strokes,
        'total_strokes': total_strokes,
        'input_frames': [img.name for img in input_images],
        'runs': []
    }
    
    for i, llm_output in enumerate(results, 1):
        output_data['runs'].append({
            'run_id': i,
            'llm_output': llm_output,
            'retry_count': retry_counts[i-1]
        })
    
    # 使用安全写入保存到文件
    output_file = output_dir / f"{stroke_name}.json"
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
        'target_stroke': target_stroke,
        'success_count': success_count,
        'retry_stats': retry_stats,
        'status': 'completed'
    }


def recognize_character(char_folder: Path, character: str,
                       num_runs: int = 5,
                       pbar: Optional[tqdm] = None) -> Dict:
    """
    识别单个字符（处理倒数第1、2、3笔）
    
    Args:
        char_folder: 字符文件夹路径
        character: 汉字
        num_runs: 每个目标笔画运行次数（默认5次）
        pbar: 进度条对象（可选）
    
    Returns:
        包含识别结果的字典
    """
    # 读取 stroke_info.json
    stroke_info_file = char_folder / "stroke_info.json"
    if not stroke_info_file.exists():
        print(f"  ⚠️ {character}: 找不到 stroke_info.json")
        return {
            'character': character,
            'status': 'missing_stroke_info'
        }
    
    try:
        with open(stroke_info_file, 'r', encoding='utf-8') as f:
            stroke_info = json.load(f)
    except Exception as e:
        print(f"  ⚠️ {character}: 读取 stroke_info.json 失败: {str(e)}")
        return {
            'character': character,
            'status': 'invalid_stroke_info'
        }
    
    stroke_count = stroke_info.get('stroke_count', len(stroke_info.get('strokes', [])))
    
    if stroke_count < 1:
        print(f"  ⚠️ {character}: 笔画数无效 ({stroke_count})")
        return {
            'character': character,
            'status': 'invalid_stroke_count',
            'stroke_count': stroke_count
        }
    
    # 创建 COF_input 文件夹
    output_dir = char_folder / "COF_input"
    output_dir.mkdir(exist_ok=True)
    
    # 确定要处理的目标笔画（倒数第1、2、3笔）
    target_strokes = []
    for offset in [1, 2, 3]:
        target_idx = stroke_count - offset + 1  # 倒数第 offset 笔
        if target_idx >= 1:  # 确保索引有效
            target_strokes.append(target_idx)
    
    if not target_strokes:
        print(f"  ⚠️ {character}: 没有可处理的笔画")
        return {
            'character': character,
            'status': 'no_valid_strokes',
            'stroke_count': stroke_count
        }
    
    # 收集重试统计
    retry_stats = {}
    processed_count = 0
    
    for target_stroke in target_strokes:
        stroke_name = f"stroke_{target_stroke:02d}"
        
        # 计算还剩几笔
        remaining_strokes = stroke_count - target_stroke
        
        # 选择输入图片
        try:
            input_images = select_frame_images(char_folder, target_stroke, stroke_count)
        except Exception as e:
            print(f"  ⚠️ {character} {stroke_name}: 选择图片失败 - {str(e)}")
            continue
        
        if not input_images:
            print(f"  ⚠️ {character} {stroke_name}: 没有可用的图片")
            continue
        
        # 运行多次识别
        results = []
        retry_counts = []
        for i in range(num_runs):
            result, retry_count = recognize_with_frames(
                input_images, character, remaining_strokes, pbar=pbar
            )
            results.append(result)
            retry_counts.append(retry_count)
            if result is None:
                pass  # 静默失败，避免干扰进度条
        
        # 构建输出数据
        output_data = {
            'character': character,
            'stroke_name': stroke_name,
            'target_stroke': target_stroke,
            'remaining_strokes': remaining_strokes,
            'total_strokes': stroke_count,
            'input_frames': [img.name for img in input_images],
            'runs': []
        }
        
        for i, llm_output in enumerate(results, 1):
            output_data['runs'].append({
                'run_id': i,
                'llm_output': llm_output,
                'retry_count': retry_counts[i-1]
            })
        
        # 使用安全写入保存到文件
        output_file = output_dir / f"{stroke_name}.json"
        safe_write_json(output_file, output_data)
        
        # 收集重试统计
        retry_stats[stroke_name] = [
            {'run_id': i+1, 'retry_count': rc} 
            for i, rc in enumerate(retry_counts)
        ]
        
        success_count = sum(1 for r in results if r is not None)
        processed_count += 1
    
    return {
        'character': character,
        'status': 'completed',
        'strokes_processed': processed_count,
        'total_strokes': stroke_count,
        'retry_stats': retry_stats
    }


def process_dataset(dataset_dir: str = "../dataset",
                   output_file: str = "cof_summary.json",
                   resume: bool = True,
                   max_workers: int = 20,
                   test_chars: Optional[int] = None,
                   skip_chars: int = 0) -> List[Dict]:
    """
    处理整个数据集（笔画级并行）
    
    Args:
        dataset_dir: 数据集目录（默认 ../dataset）
        output_file: 输出摘要文件
        resume: 是否断点续传
        max_workers: 并行线程数
        test_chars: 测试字符数（None 表示处理全部）
        skip_chars: 跳过前 N 个字符
    
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
    
    # 测试模式：跳过和限制
    if skip_chars > 0:
        char_folders = char_folders[skip_chars:]
    if test_chars:
        char_folders = char_folders[:test_chars]
    
    print(f"\n{'='*60}")
    print(f"🎯 Chain-of-Frames 输入识别（笔画级并行版）")
    print(f"{'='*60}")
    print(f"数据集: {dataset_path}")
    print(f"字符数: {len(char_folders)}")
    print(f"并行数: {max_workers}")
    print(f"处理策略: 每个字符的倒数第1、2、3笔")
    print(f"输入采样: 每笔均匀采样5帧")
    print(f"并行粒度: 笔画级（每个线程处理一笔）")
    print(f"输出位置: 各字符的 COF_input/ 文件夹")
    
    # 确定重试统计文件路径
    retry_stats_path = script_dir / output_file.replace('.json', '_retry.json')
    
    # 加载已有的重试统计（如果是 resume 模式）
    all_retry_stats = {}
    if resume and retry_stats_path.exists():
        try:
            with open(retry_stats_path, 'r', encoding='utf-8') as f:
                all_retry_stats = json.load(f)
        except:
            pass
    
    # ✅ 笔画级并行：收集所有待处理的笔画任务
    stroke_tasks = []
    char_to_strokes = {}  # 记录每个字符对应的笔画任务索引
    
    for char_folder in char_folders:
        character = char_folder.name
        
        # 读取 stroke_info 确定总笔画数
        stroke_info_file = char_folder / "stroke_info.json"
        if not stroke_info_file.exists():
            continue
        
        try:
            with open(stroke_info_file, 'r', encoding='utf-8') as f:
                stroke_info = json.load(f)
            stroke_count = stroke_info.get('stroke_count', len(stroke_info.get('strokes', [])))
        except:
            continue
        
        if stroke_count < 1:
            continue
        
        # 创建 COF_input 文件夹
        output_dir = char_folder / "COF_input"
        output_dir.mkdir(exist_ok=True)
        
        char_to_strokes[character] = []
        
        # 确定要处理的目标笔画（倒数第1、2、3笔）
        for offset in [1, 2, 3]:
            target_idx = stroke_count - offset + 1  # 倒数第 offset 笔
            if target_idx < 1:
                continue
            
            # 检查断点续传：该笔画是否已完成
            stroke_file = output_dir / f"stroke_{target_idx:02d}.json"
            if resume and is_stroke_file_valid(stroke_file, expected_runs=5):
                continue  # 跳过已完成的笔画
            
            # 添加笔画任务
            task_idx = len(stroke_tasks)
            stroke_tasks.append({
                'char_folder': char_folder,
                'character': character,
                'target_stroke': target_idx,
                'total_strokes': stroke_count
            })
            char_to_strokes[character].append(task_idx)
    
    print(f"待处理笔画任务: {len(stroke_tasks)} 个")
    print(f"{'='*60}\n")
    
    if len(stroke_tasks) == 0:
        print("✅ 所有笔画都已处理完成！")
        return []
    
    # 计算总的 API 调用次数（每个笔画5次）
    total_api_calls = len(stroke_tasks) * 5
    
    print(f"预计 API 调用: {total_api_calls} 次")
    print(f"{'='*60}\n")
    
    results = []
    start_time = time.time()
    
    # 笔画级并行处理
    stroke_results = [None] * len(stroke_tasks)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=total_api_calls, desc="API 调用", unit="次", 
                  ncols=100, smoothing=0.1) as pbar:
            future_to_idx = {
                executor.submit(
                    recognize_single_stroke,
                    task['char_folder'],
                    task['character'],
                    task['target_stroke'],
                    task['total_strokes'],
                    5,  # num_runs
                    pbar
                ): idx
                for idx, task in enumerate(stroke_tasks)
            }
            
            completed_strokes = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    stroke_results[idx] = result
                    
                    # 保存重试统计
                    if result and 'retry_stats' in result:
                        char_name = result['character']
                        if char_name not in all_retry_stats:
                            all_retry_stats[char_name] = {}
                        all_retry_stats[char_name].update(result['retry_stats'])
                    
                    completed_strokes += 1
                    
                    # 每10个笔画任务保存一次
                    if completed_strokes % 10 == 0:
                        # 汇总字符级结果
                        char_summary = {}
                        for char_name, task_indices in char_to_strokes.items():
                            strokes_processed = sum(
                                1 for ti in task_indices 
                                if ti < len(stroke_results) and stroke_results[ti] is not None
                            )
                            if strokes_processed > 0 or len(task_indices) > 0:
                                char_summary[char_name] = strokes_processed
                        
                        results_sorted = [
                            {'character': char, 'strokes_processed': count}
                            for char, count in sorted(char_summary.items())
                        ]
                        
                        summary_data = {
                            'total_characters': len(char_folders),
                            'processed_strokes': completed_strokes,
                            'total_stroke_tasks': len(stroke_tasks),
                            'processing_time_seconds': time.time() - start_time,
                            'results': results_sorted
                        }
                        safe_write_json(output_path, summary_data)
                        
                        # 保存重试统计
                        if all_retry_stats:
                            safe_write_json(retry_stats_path, all_retry_stats)
                        
                        pbar.set_postfix_str(f"已完成 {completed_strokes}/{len(stroke_tasks)} 笔画")
                    
                except Exception as e:
                    print(f"\n❌ 笔画处理失败: {str(e)[:100]}")
                    stroke_results[idx] = None
    
    total_time = time.time() - start_time
    
    # 最终汇总：按字符组织
    char_summary = {}
    for char_name, task_indices in char_to_strokes.items():
        strokes_processed = sum(
            1 for ti in task_indices 
            if ti < len(stroke_results) and stroke_results[ti] is not None
        )
        
        # 读取总笔画数
        char_folder = None
        for cf in char_folders:
            if cf.name == char_name:
                char_folder = cf
                break
        
        total_strokes = 0
        if char_folder:
            stroke_info_file = char_folder / "stroke_info.json"
            if stroke_info_file.exists():
                try:
                    with open(stroke_info_file, 'r', encoding='utf-8') as f:
                        stroke_info = json.load(f)
                    total_strokes = stroke_info.get('stroke_count', 0)
                except:
                    pass
        
        char_summary[char_name] = {
            'character': char_name,
            'strokes_processed': strokes_processed,
            'total_strokes': total_strokes,
            'status': 'completed' if strokes_processed > 0 else 'failed'
        }
    
    results_sorted = sorted(char_summary.values(), key=lambda x: x['character'])
    
    # 最终保存摘要
    summary_data = {
        'total_characters': len(char_folders),
        'processed_strokes': len([r for r in stroke_results if r is not None]),
        'total_stroke_tasks': len(stroke_tasks),
        'processing_time_seconds': total_time,
        'results': results_sorted
    }
    safe_write_json(output_path, summary_data)
    
    # 最终确认保存重试统计
    if all_retry_stats:
        safe_write_json(retry_stats_path, all_retry_stats)
        print(f"\n📊 重试统计已保存到: {retry_stats_path}")
    
    print(f"\n{'='*60}")
    print(f"✅ 完成！")
    print(f"{'='*60}")
    print(f"耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
    print(f"处理: {len(stroke_tasks)} 个笔画任务")
    print(f"总计: {len(char_folders)} 个字符")
    print(f"摘要文件: {output_path}")
    print(f"识别结果: 各字符的 COF_input/ 文件夹")
    print(f"重试统计: {retry_stats_path}")
    print(f"{'='*60}\n")
    
    return results_sorted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chain-of-Frames 输入识别（增强版，处理倒数三笔）"
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
        default="cof_summary.json",
        help="输出摘要文件 (默认: cof_summary.json)"
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
        process_dataset(
            dataset_dir=args.dataset,
            output_file=args.output,
            resume=not args.no_resume,
            max_workers=args.max_workers,
            test_chars=args.test,
            skip_chars=args.skip
        )
    except KeyboardInterrupt:
        # 捕获 Ctrl+C 中断
        print_speed_stats()
        sys.exit(0)
