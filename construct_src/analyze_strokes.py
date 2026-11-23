#!/usr/bin/env python3
"""
使用大模型识别汉字笔画数和笔画顺序
"""

import os
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from typing import Dict, List, Optional

# API配置
API_URL = "https://llm.q2zy.com/v1/chat/completions"
API_KEY = os.environ.get("MUSO_API_KEY", "")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}


def recognize_stroke_info(character: str, max_retries: int = 15, initial_delay: float = 0.5) -> Optional[Dict]:
    """
    识别单个汉字的笔画信息（带自动重试）
    
    Args:
        character: 要识别的汉字
        max_retries: 最大重试次数（默认15次）
        initial_delay: 初始重试延迟（秒）
    
    Returns:
        包含笔画信息的字典，失败返回None
    """
    
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
        "temperature": 1.0 # 降低温度以获得更一致的结果
    }
    
    for retry in range(max_retries):
        try:
            response = requests.post(API_URL, headers=HEADERS, json=data, timeout=30)
            
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
                
                # 解析JSON
                try:
                    stroke_data = json.loads(content)
                    
                    # 验证必要字段
                    required_fields = ['character', 'stroke_count', 'strokes']
                    if all(field in stroke_data for field in required_fields):
                        stroke_data['success'] = True
                        return stroke_data
                    else:
                        print(f"    警告: JSON缺少必要字段")
                        if retry < max_retries - 1:
                            delay = initial_delay * (2 ** retry)
                            time.sleep(min(delay, 5))
                            continue
                        return None
                        
                except json.JSONDecodeError as e:
                    print(f"    警告: JSON解析失败 - {str(e)[:50]}")
                    if retry < max_retries - 1:
                        delay = initial_delay * (2 ** retry)
                        time.sleep(min(delay, 5))
                        continue
                    return None
            else:
                print(f"    重试 {retry + 1}/{max_retries}: 状态码 {response.status_code}")
                delay = initial_delay * (2 ** min(retry, 5))
                time.sleep(min(delay, 5))
                
        except requests.exceptions.Timeout:
            print(f"    重试 {retry + 1}/{max_retries}: 请求超时")
            delay = initial_delay * (2 ** min(retry, 5))
            time.sleep(min(delay, 5))
        except Exception as e:
            print(f"    重试 {retry + 1}/{max_retries}: 错误 - {str(e)[:50]}")
            delay = initial_delay * (2 ** min(retry, 5))
            time.sleep(min(delay, 5))
    
    print(f"    ❌ 所有重试都失败了")
    return None


def analyze_character_simple(character: str) -> Optional[Dict]:
    """
    分析单个汉字（简化版，不额外重试）
    
    Args:
        character: 要分析的汉字
    
    Returns:
        笔画信息，失败返回None
    """
    print(f"正在分析汉字: '{character}'", end=" ", flush=True)
    
    start_time = time.time()
    
    # 直接调用，依赖 recognize_stroke_info 的重试机制
    result = recognize_stroke_info(character, max_retries=20, initial_delay=0.5)
    
    elapsed_time = time.time() - start_time
    
    if result and result.get('success'):
        print(f"✅ (耗时: {elapsed_time:.1f}秒)")
        print(f"  笔画数: {result['stroke_count']}, 笔画: {', '.join(result['strokes'][:5])}{'...' if len(result['strokes']) > 5 else ''}")
        return result
    else:
        print(f"❌ 失败 (耗时: {elapsed_time:.1f}秒)")
        return None


def analyze_single_char_wrapper(char_data: tuple) -> Dict:
    """
    包装函数，用于并行处理
    
    Args:
        char_data: (索引, 总数, 字符, 文件夹路径)
    
    Returns:
        分析结果字典
    """
    idx, total, character, char_folder = char_data
    
    print(f"[{idx}/{total}] ", end="", flush=True)
    
    stroke_info = analyze_character_simple(character)
    
    if stroke_info:
        # 统计该字有多少张图片
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


def analyze_dataset_characters(dataset_dir: str = "../dataset", 
                               max_workers: int = 3,
                               save_interval: int = 10,
                               resume: bool = True) -> List[Dict]:
    """
    并行分析数据集中所有汉字的笔画信息（支持断点续传）
    
    Args:
        dataset_dir: 数据集目录（相对于脚本目录）
        max_workers: 并行处理的字数
        save_interval: 每处理多少个字保存一次结果
        resume: 是否从上次中断处继续
    
    Returns:
        所有汉字的笔画信息列表
    """
    # 解析路径
    script_dir = Path(__file__).parent
    dataset_path = script_dir / dataset_dir
    output_file = script_dir.parent / "stroke_analysis_results.json"
    
    if not dataset_path.exists():
        print(f"❌ 错误: 数据集目录不存在 - {dataset_path}")
        return []
    
    # 获取所有汉字文件夹
    char_folders = sorted([
        d for d in dataset_path.iterdir() 
        if d.is_dir() and len(d.name) > 0
    ])
    
    # 过滤掉非汉字文件夹（如 summary.txt）
    char_folders = [d for d in char_folders if '\u4e00' <= d.name[0] <= '\u9fff']
    all_characters = [d.name for d in char_folders]
    
    # 检查是否有已存在的结果（用于resume）
    existing_results = {}
    if resume and output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                saved_results = json.load(f)
                for result in saved_results:
                    existing_results[result['character']] = result
            print(f"📂 发现已有结果文件，已完成 {len(existing_results)} 个字")
        except:
            print(f"⚠️  无法读取已有结果文件，将重新开始")
    
    # 筛选出还需要处理的字
    remaining_folders = [
        d for d in char_folders 
        if d.name not in existing_results
    ]
    
    print(f"\n{'='*60}")
    print(f"数据集总字数: {len(char_folders)}")
    print(f"已完成: {len(existing_results)}")
    print(f"待处理: {len(remaining_folders)}")
    print(f"同时处理 {max_workers} 个字，每个字失败时最多重试15次")
    print(f"{'='*60}\n")
    
    if len(remaining_folders) == 0:
        print("✅ 所有字符都已处理完成！")
        return list(existing_results.values())
    
    # 从已有结果开始
    results = list(existing_results.values())
    start_time = time.time()
    
    # 准备任务数据（只处理剩余的）
    total_chars = len(char_folders)
    already_done = len(existing_results)
    tasks = [
        (already_done + idx, total_chars, char_folder.name, char_folder)
        for idx, char_folder in enumerate(remaining_folders, 1)
    ]
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(analyze_single_char_wrapper, task): task[0]
            for task in tasks
        }
        
        # 收集结果（按完成顺序）
        completed_count = already_done
        for future in as_completed(future_to_idx):
            result = future.result()
            results.append(result)
            completed_count += 1
            
            # 定期保存结果
            if (completed_count - already_done) % save_interval == 0 or completed_count == total_chars:
                # 按字符顺序排序结果
                results_sorted = sorted(results, key=lambda x: x['character'])
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results_sorted, f, ensure_ascii=False, indent=2)
                print(f"\n  💾 已保存 {completed_count}/{total_chars} 个结果到 {output_file}")
                print("-" * 60)
    
    total_time = time.time() - start_time
    success_count = sum(1 for r in results if 'stroke_count' in r)
    
    # 最终排序保存
    results_sorted = sorted(results, key=lambda x: x['character'])
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_sorted, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ 全部完成！")
    print(f"  本次耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
    print(f"  本次处理: {len(remaining_folders)} 个字")
    print(f"  总计完成: {len(results)}/{total_chars}")
    print(f"  成功: {success_count}")
    print(f"  失败: {len(results) - success_count}")
    if len(remaining_folders) > 0:
        print(f"  平均每字: {total_time/len(remaining_folders):.2f}秒")
    print(f"  结果已保存到: {output_file}")
    print(f"{'='*60}\n")
    
    return results_sorted


def print_summary(results: List[Dict]):
    """打印分析结果摘要"""
    if not results:
        print("没有结果可显示")
        return
    
    successful_results = [r for r in results if 'stroke_count' in r]
    
    if not successful_results:
        print("没有成功的分析结果")
        return
    
    print(f"\n{'='*60}")
    print("笔画统计摘要")
    print(f"{'='*60}\n")
    
    # 按笔画数分组
    stroke_groups = {}
    for r in successful_results:
        count = r['stroke_count']
        if count not in stroke_groups:
            stroke_groups[count] = []
        stroke_groups[count].append(r['character'])
    
    # 打印统计
    for stroke_count in sorted(stroke_groups.keys()):
        chars = stroke_groups[stroke_count]
        print(f"{stroke_count:2d}画 ({len(chars):2d}字): {' '.join(chars[:20])}")
        if len(chars) > 20:
            print(f"      ... 等共{len(chars)}字")
    
    print(f"\n笔画数范围: {min(stroke_groups.keys())} - {max(stroke_groups.keys())}画")
    print(f"平均笔画数: {sum(r['stroke_count'] for r in successful_results) / len(successful_results):.1f}画")


# ============= 使用示例 =============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="分析汉字笔画信息")
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="../dataset",
        help="数据集目录 (默认: ../dataset，相对于脚本目录)"
    )
    parser.add_argument(
        "-w", "--max-workers",
        type=int,
        default=3,
        help="并行处理的字数 (默认: 3)"
    )
    parser.add_argument(
        "-s", "--save-interval",
        type=int,
        default=10,
        help="保存间隔 (默认: 每10个字)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="不使用断点续传，重新开始"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="测试模式：只分析前5个字"
    )
    
    args = parser.parse_args()
    
    # 解析路径
    script_dir = Path(__file__).parent
    dataset_path = script_dir / args.dataset
    
    if args.test:
        print("\n⚠️  测试模式：只分析前5个字\n")
        
        if not dataset_path.exists():
            print(f"❌ 错误: 数据集目录不存在 - {dataset_path}")
            exit(1)
        
        char_folders = sorted([
            d for d in dataset_path.iterdir() 
            if d.is_dir() and len(d.name) > 0 and '\u4e00' <= d.name[0] <= '\u9fff'
        ])[:5]
        
        # 准备任务
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
        
        # 排序并保存测试结果
        results_sorted = sorted(results, key=lambda x: x['character'])
        test_output = script_dir.parent / "stroke_analysis_test.json"
        with open(test_output, 'w', encoding='utf-8') as f:
            json.dump(results_sorted, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 测试结果已保存到: {test_output}")
        
        print_summary(results_sorted)
    else:
        # 完整分析
        results = analyze_dataset_characters(
            dataset_dir=args.dataset,
            max_workers=args.max_workers,
            save_interval=args.save_interval,
            resume=not args.no_resume  # 默认开启resume
        )
        
        print_summary(results)
