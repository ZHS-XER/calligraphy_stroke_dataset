#!/usr/bin/env python3
"""
Chain-of-Frames 输入识别脚本

从 stroke_01 到 HERP-1，均匀采样4张图片，加上 HERP 图片，共5张图片作为输入
结果保存到各字符的 COF_input/ 文件夹
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

# API配置
API_URL = "https://llm.q2zy.com/v1/chat/completions"
API_KEY = "MUSO_API_KEY_REMOVED"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
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


def select_frame_images(char_folder: Path, herp: int) -> List[Path]:
    """
    选择输入帧图片
    
    从 stroke_01 到 stroke_{HERP-1}，尽量均匀采样4张，加上 stroke_{HERP}，共5张
    
    Args:
        char_folder: 字符文件夹路径
        herp: HERP 值（人类最早识别点的笔画索引，从1开始）
    
    Returns:
        选中的图片路径列表（按笔画顺序）
    """
    # HERP 图片（必选）
    herp_image = char_folder / f"stroke_{herp:02d}.png"
    if not herp_image.exists():
        raise FileNotFoundError(f"HERP 图片不存在: {herp_image}")
    
    # 候选图片: stroke_01 到 stroke_{HERP-1}
    candidate_range = list(range(1, herp))  # [1, 2, ..., HERP-1]
    
    if len(candidate_range) == 0:
        # 如果 HERP = 1，只返回 HERP 图片
        return [herp_image]
    
    # 选择策略：尽量均匀采样4张
    if len(candidate_range) <= 4:
        # 不足4张，全部选择
        selected_indices = candidate_range
    else:
        # 均匀采样4张
        step = (len(candidate_range) - 1) / 3  # 分成4段
        selected_indices = [
            candidate_range[0],  # 第一张
            candidate_range[round(step)],
            candidate_range[round(2 * step)],
            candidate_range[-1]  # 最后一张（HERP-1）
        ]
        # 去重（以防 round 导致重复）
        selected_indices = sorted(list(set(selected_indices)))
    
    # 构建图片路径列表
    selected_images = []
    for idx in selected_indices:
        img_path = char_folder / f"stroke_{idx:02d}.png"
        if img_path.exists():
            selected_images.append(img_path)
    
    # 添加 HERP 图片
    selected_images.append(herp_image)
    
    return selected_images


def recognize_with_frames(images: List[Path], character: str, remaining_strokes: int,
                         max_retries: int = 300, initial_delay: float = 2.0,
                         pbar: Optional[tqdm] = None) -> Optional[Dict]:
    """
    使用多帧图片进行识别（带增强重试机制）
    
    Args:
        images: 图片路径列表（按笔画顺序）
        character: 汉字（用于日志）
        remaining_strokes: HERP 之后还剩几个笔画
        max_retries: 最大重试次数（默认300次）
        initial_delay: 初始重试延迟（默认2秒）
        pbar: 进度条对象（可选）
    
    Returns:
        完整的 API JSON 响应，失败返回 None
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
        return None
    
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
                    if pbar:
                        pbar.update(1)  # 成功完成一次 API 调用
                    return llm_output
                        
                except json.JSONDecodeError as e:
                    print(f"    警告: JSON 解析失败 - {str(e)[:50]}")
                    if retry < max_retries - 1:
                        delay = initial_delay * (2 ** min(retry, 4))
                        time.sleep(min(delay, 30))
                        continue
                    return None
            else:
                if retry < max_retries - 1:
                    delay = initial_delay * (2 ** min(retry, 4))
                    wait_time = min(delay, 30)
                    time.sleep(wait_time)
                
        except requests.exceptions.Timeout:
            if retry < max_retries - 1:
                delay = initial_delay * (2 ** min(retry, 4))
                wait_time = min(delay, 30)
                time.sleep(wait_time)
        except Exception as e:
            if retry < max_retries - 1:
                delay = initial_delay * (2 ** min(retry, 4))
                wait_time = min(delay, 30)
                time.sleep(wait_time)
    
    return None


def recognize_character(char_folder: Path, character: str,
                       num_runs: int = 5,
                       pbar: Optional[tqdm] = None) -> Dict:
    """
    识别单个字符（使用 Chain-of-Frames 输入）
    
    Args:
        char_folder: 字符文件夹路径
        character: 汉字
        num_runs: 运行次数（默认5次）
        pbar: 进度条对象（可选）
    
    Returns:
        包含识别结果的字典
    """
    # 读取 stroke_info.json
    stroke_info_file = char_folder / "stroke_info.json"
    if not stroke_info_file.exists():
        print(f"  ⚠️ 找不到 stroke_info.json")
        return {
            'character': character,
            'status': 'missing_stroke_info'
        }
    
    with open(stroke_info_file, 'r', encoding='utf-8') as f:
        stroke_info = json.load(f)
    
    herp = stroke_info.get('HERP')
    if herp is None:
        print(f"  ⚠️ stroke_info.json 中没有 HERP 字段")
        return {
            'character': character,
            'status': 'missing_herp'
        }
    
    stroke_count = stroke_info.get('stroke_count', len(stroke_info.get('strokes', [])))
    remaining_strokes = stroke_count - herp
    
    # 选择输入图片
    try:
        input_images = select_frame_images(char_folder, herp)
    except Exception as e:
        print(f"  ⚠️ 选择图片失败: {str(e)}")
        return {
            'character': character,
            'status': 'image_selection_failed',
            'error': str(e)
        }
    
    print(f"  字符: {character}, HERP: {herp}, 剩余笔画: {remaining_strokes}")
    print(f"  输入帧: {[img.name for img in input_images]}")
    
    # 创建 COF_input 文件夹
    output_dir = char_folder / "COF_input"
    output_dir.mkdir(exist_ok=True)
    
    # 运行多次识别
    results = []
    for i in range(num_runs):
        result = recognize_with_frames(input_images, character, remaining_strokes, pbar=pbar)
        results.append(result)
        if result is None:
            print(f"    ⚠️ 第 {i+1} 次识别失败")
    
    # 保存结果
    output_data = {
        'character': character,
        'herp': herp,
        'remaining_strokes': remaining_strokes,
        'total_strokes': stroke_count,
        'input_frames': [img.name for img in input_images],
        'runs': []
    }
    
    for i, llm_output in enumerate(results, 1):
        output_data['runs'].append({
            'run_id': i,
            'llm_output': llm_output
        })
    
    # 保存到文件
    output_file = output_dir / "recognition_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    success_count = sum(1 for r in results if r is not None)
    print(f"  ✅ 完成 ({success_count}/{num_runs})")
    
    return {
        'character': character,
        'status': 'completed',
        'success_count': success_count,
        'total_runs': num_runs
    }


def process_dataset(dataset_dir: str = "../test_dataset",
                   output_file: str = "cof_summary.json",
                   resume: bool = True,
                   max_workers: int = 20,
                   test_chars: Optional[int] = None) -> List[Dict]:
    """
    处理整个数据集
    
    Args:
        dataset_dir: 数据集目录
        output_file: 输出摘要文件
        resume: 是否断点续传
        max_workers: 并行线程数
        test_chars: 测试字符数（None 表示处理全部）
    
    Returns:
        所有字符的处理摘要列表
    """
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
    
    # 测试模式
    if test_chars:
        char_folders = char_folders[:test_chars]
    
    print(f"\n{'='*60}")
    print(f"🎯 Chain-of-Frames 输入识别")
    print(f"{'='*60}")
    print(f"数据集: {dataset_path}")
    print(f"字符数: {len(char_folders)}")
    print(f"并行数: {max_workers}")
    print(f"输出位置: 各字符的 COF_input/ 文件夹")
    
    # 断点续传：检查哪些字符已经处理完成
    processed_chars = set()
    if resume:
        for char_folder in char_folders:
            output_dir = char_folder / "COF_input"
            output_file_path = output_dir / "recognition_result.json"
            if output_file_path.exists():
                try:
                    with open(output_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 检查是否完成了所有运行
                        if len(data.get('runs', [])) >= 5:
                            processed_chars.add(char_folder.name)
                except:
                    pass
        
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
    
    # 计算总的 API 调用次数（每个字符5次）
    total_api_calls = len(remaining_folders) * 5
    
    print(f"预计 API 调用: {total_api_calls} 次")
    print(f"{'='*60}\n")
    
    results = []
    start_time = time.time()
    
    # 并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=total_api_calls, desc="API 调用", unit="次", 
                  ncols=100, smoothing=0.1) as pbar:
            future_to_char = {
                executor.submit(recognize_character, char_folder, char_folder.name, pbar=pbar): char_folder.name
                for char_folder in remaining_folders
            }
            
            completed_count = len(processed_chars)
            for future in as_completed(future_to_char):
                result = future.result()
                results.append(result)
                completed_count += 1
                
                # 每10个字保存一次摘要
                if completed_count % 10 == 0 or completed_count == len(char_folders):
                    results_sorted = sorted(results, key=lambda x: x['character'])
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'total_characters': len(char_folders),
                            'processed_characters': completed_count,
                            'results': results_sorted
                        }, f, ensure_ascii=False, indent=2)
                    pbar.set_postfix_str(f"已完成 {completed_count}/{len(char_folders)} 字")
    
    total_time = time.time() - start_time
    
    # 最终保存摘要
    results_sorted = sorted(results, key=lambda x: x['character'])
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_characters': len(char_folders),
            'processed_characters': completed_count,
            'processing_time_seconds': total_time,
            'results': results_sorted
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ 完成！")
    print(f"{'='*60}")
    print(f"耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
    print(f"处理: {len(remaining_folders)} 个字")
    print(f"总计: {completed_count}/{len(char_folders)}")
    print(f"摘要文件: {output_path}")
    print(f"识别结果: 各字符的 COF_input/ 文件夹")
    print(f"{'='*60}\n")
    
    return results_sorted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chain-of-Frames 输入识别（使用 test_dataset）"
    )
    parser.add_argument("-d", "--dataset", default="../test_dataset",
                       help="数据集目录路径（默认: ../test_dataset）")
    parser.add_argument("-o", "--output", default="cof_summary.json",
                       help="输出摘要文件名（默认: cof_summary.json）")
    parser.add_argument("-w", "--workers", type=int, default=20,
                       help="并行线程数（默认: 20）")
    parser.add_argument("--no-resume", action="store_true",
                       help="不使用断点续传（重新处理所有字符）")
    parser.add_argument("--test", type=int, metavar="N",
                       help="测试模式：只处理前 N 个字符")
    
    args = parser.parse_args()
    
    process_dataset(
        dataset_dir=args.dataset,
        output_file=args.output,
        resume=not args.no_resume,
        max_workers=args.workers,
        test_chars=args.test
    )
