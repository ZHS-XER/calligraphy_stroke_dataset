#!/usr/bin/env python3
"""
识别 dataset 中每个字的笔画图片

纯粹收集大模型的 JSON 输出，保存到一个大文件
支持断点续传和失败重试
"""

import os
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
API_KEY = os.environ.get("MUSO_API_KEY", "")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# 加载 prompt 模板
PROMPT_TEMPLATE = """任务：识别单帧书法图像中的汉字（该帧信息已足够唯一确定该字）。

上下文：
- 本图像是汉字书法教学视频的截图。
- 当前帧包含足以判断目标汉字的笔画与结构。

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


def is_stroke_file_valid(file_path: Path) -> bool:
    """
    验证笔画识别结果文件是否有效
    
    Args:
        file_path: JSON文件路径
    
    Returns:
        文件是否有效（能解析且所有run的llm_output有效）
    """
    if not file_path.exists():
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查是否有runs字段
        if 'runs' not in data or not isinstance(data['runs'], list):
            return False
        
        # 检查是否有足够的runs（应该是5个）
        if len(data['runs']) != 5:
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
                                max_retries: int = 300, initial_delay: float = 2.0,
                                pbar: Optional[tqdm] = None) -> Optional[Dict]:
    """
    识别单个笔画图片一次（带增强重试机制）
    
    Args:
        image_path: 图片路径
        character: 汉字（用于日志）
        max_retries: 最大重试次数（默认300次）
        initial_delay: 初始重试延迟（默认2秒）
        pbar: 进度条对象（可选）
    
    Returns:
        完整的 API JSON 响应，失败返回 None
    """
    try:
        image_base64 = image_to_base64(image_path)
    except Exception as e:
        print(f"    ❌ 读取图片失败: {str(e)[:50]}")
        return None
    
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
                    {"type": "text", "text": PROMPT_TEMPLATE},
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
                    if pbar:
                        pbar.update(1)  # 成功完成一次 API 调用
                    return llm_output  # 直接返回大模型的 JSON 输出
                        
                except json.JSONDecodeError as e:
                    print(f"    警告: JSON 解析失败 - {str(e)[:50]}")
                    if retry < max_retries - 1:
                        # 指数退避，但最大等待30秒
                        delay = initial_delay * (2 ** min(retry, 4))
                        time.sleep(min(delay, 30))
                        continue
                    return None
            else:
                if retry < max_retries - 1:
                    # 指数退避，最大等待30秒
                    delay = initial_delay * (2 ** min(retry, 4))
                    wait_time = min(delay, 30)
                    time.sleep(wait_time)
                
        except requests.exceptions.Timeout:
            if retry < max_retries - 1:
                # 超时后等待更久
                delay = initial_delay * (2 ** min(retry, 4))
                wait_time = min(delay, 30)
                time.sleep(wait_time)
        except Exception as e:
            if retry < max_retries - 1:
                # 其他错误也增加等待时间
                delay = initial_delay * (2 ** min(retry, 4))
                wait_time = min(delay, 30)
                time.sleep(wait_time)
    
    return None


def recognize_stroke_image_multiple(image_path: Path, character: str, 
                                    num_runs: int = 5,
                                    pbar: Optional[tqdm] = None) -> List[Optional[Dict]]:
    """
    对单个笔画图片运行多次识别
    
    Args:
        image_path: 图片路径
        character: 汉字（用于日志）
        num_runs: 运行次数
        pbar: 进度条对象（可选）
    
    Returns:
        包含多次识别结果的列表
    """
    results = []
    for i in range(num_runs):
        result = recognize_stroke_image_once(image_path, character, pbar=pbar)
        results.append(result)
        if result is None:
            print(f"    ⚠️ 第 {i+1} 次识别失败")
    return results


def get_stroke_images(char_folder: Path) -> List[Path]:
    """
    获取字符文件夹中的笔画图片（只返回最后5个，排除整字图）
    
    Args:
        char_folder: 字符文件夹路径
    
    Returns:
        笔画图片路径列表（最后5个）
    """
    stroke_files = sorted([
        f for f in char_folder.glob("stroke_*.png")
        if f.name != "stroke_00.png"  # 排除整字图
    ])
    # 只返回最后5个笔画
    return stroke_files[-5:] if len(stroke_files) > 5 else stroke_files


def recognize_character(char_folder: Path, character: str,
                       pbar: Optional[tqdm] = None) -> Dict:
    """
    识别单个字符的笔画（只处理最后5个笔画，每张图跑5次）
    
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
    
    # 创建 recognition 文件夹
    recognition_dir = char_folder / "recognition"
    recognition_dir.mkdir(exist_ok=True)
    
    # 逐个识别（每张图跑5次）
    processed_count = 0
    for stroke_file in stroke_images:
        stroke_name = stroke_file.stem
        # print(f"  识别 {stroke_name} (5次)...", end=" ", flush=True)  # 注释掉
        
        # 运行5次识别，传递进度条
        results = recognize_stroke_image_multiple(stroke_file, character, num_runs=5, pbar=pbar)
        
        # 保存到单独的 JSON 文件
        output_data = {
            'character': character,
            'stroke_name': stroke_name,
            'image_path': str(stroke_file.relative_to(char_folder.parent.parent)),
            'runs': []
        }
        
        for i, llm_output in enumerate(results, 1):
            output_data['runs'].append({
                'run_id': i,
                'llm_output': llm_output
            })
        
        # 保存到文件
        output_file = recognition_dir / f"{stroke_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        success_count = sum(1 for r in results if r is not None)
        # print(f"✅ ({success_count}/5)")  # 注释掉
        processed_count += 1
    
    return {
        'character': character,
        'strokes_processed': processed_count
    }


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
    print(f"🎯 收集大模型识别结果（只处理最后5个笔画，每张图5次）")
    print(f"{'='*60}")
    print(f"数据集: {dataset_path}")
    print(f"字符数: {len(char_folders)}")
    print(f"并行数: {max_workers}")
    print(f"输出位置: 每个字符的 recognition/ 文件夹")
    
    # 断点续传：检查哪些字符已经处理完成
    processed_chars = set()
    incomplete_chars = []  # 记录需要重新处理的字符
    
    if resume:
        for char_folder in char_folders:
            recognition_dir = char_folder / "recognition"
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
                elif len(expected_files) > 0:
                    # 有文件但不完整，记录下来
                    incomplete_chars.append(char_folder.name)
        
        if processed_chars:
            print(f"📂 已完成 {len(processed_chars)} 个字")
        if incomplete_chars:
            print(f"⚠️  检测到 {len(incomplete_chars)} 个字符文件不完整，将重新处理")
            # 只显示前10个作为示例
            if len(incomplete_chars) <= 10:
                print(f"    {', '.join(incomplete_chars)}")
            else:
                print(f"    {', '.join(incomplete_chars[:10])} ...")
    
    remaining_folders = [
        d for d in char_folders
        if d.name not in processed_chars
    ]
    
    print(f"待处理: {len(remaining_folders)}")
    print(f"{'='*60}\n")
    
    if len(remaining_folders) == 0:
        print("✅ 所有字符都已处理完成！")
        return []
    
    # 计算总的 API 调用次数（每个字符最多5个笔画，每个笔画5次）
    total_api_calls = 0
    for char_folder in remaining_folders:
        stroke_images = get_stroke_images(char_folder)
        total_api_calls += len(stroke_images) * 5  # 每个笔画5次
    
    print(f"预计 API 调用: {total_api_calls} 次")
    print(f"{'='*60}\n")
    
    results = []
    start_time = time.time()
    
    # 并行处理，使用 API 调用级别的进度条
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建进度条，以 API 调用次数为单位
        with tqdm(total=total_api_calls, desc="API 调用", unit="次", 
                  ncols=100, smoothing=0.1) as pbar:
            future_to_char = {
                executor.submit(recognize_character, char_folder, char_folder.name, pbar): char_folder.name
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
    print(f"识别结果: 各字符的 recognition/ 文件夹")
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
        help="测试模式：只处理前 N 个字符"
    )
    
    args = parser.parse_args()
    
    if args.test:
        print(f"\n⚠️  测试模式：只处理前 {args.test} 个字符\n")
        
        script_dir = Path(__file__).parent
        dataset_path = script_dir / args.dataset
        
        char_folders = sorted([
            d for d in dataset_path.iterdir()
            if d.is_dir() and len(d.name) > 0 and '\u4e00' <= d.name[0] <= '\u9fff'
        ])[:args.test]
        
        results = []
        for char_folder in char_folders:
            result = recognize_character(char_folder, char_folder.name)
            results.append(result)
        
        test_output = script_dir / f"test_{args.test}.json"
        with open(test_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 测试结果: {test_output}")
    else:
        process_dataset(
            dataset_dir=args.dataset,
            output_file=args.output,
            resume=not args.no_resume,
            max_workers=args.max_workers
        )
