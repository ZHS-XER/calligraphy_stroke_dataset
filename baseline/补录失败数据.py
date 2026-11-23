#!/usr/bin/env python3
"""
补录失败的识别数据

扫描所有 recognition/*.json 文件，找出 llm_output 为 null 的条目，
重新进行 API 调用，更新 JSON 文件
"""

import os
import requests
import base64
import json
from pathlib import Path
import time
from typing import Dict, Optional
from tqdm import tqdm
import argparse

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


def recognize_image_once(image_path: Path, max_retries: int = 100, 
                        initial_delay: float = 0.5) -> Optional[Dict]:
    """
    识别单个图片一次（带重试机制）
    """
    try:
        image_base64 = image_to_base64(image_path)
    except Exception as e:
        print(f"读取图片失败: {image_path}: {str(e)[:50]}")
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
            response = requests.post(API_URL, headers=HEADERS, json=data, timeout=60)
            
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
                    return llm_output
                        
                except json.JSONDecodeError as e:
                    if retry < max_retries - 1:
                        delay = initial_delay * (2 ** retry)
                        time.sleep(min(delay, 10))
                        continue
                    return None
            else:
                if retry < max_retries - 1:
                    delay = initial_delay * (2 ** min(retry, 5))
                    time.sleep(min(delay, 10))
                
        except requests.exceptions.Timeout:
            if retry < max_retries - 1:
                delay = initial_delay * (2 ** min(retry, 5))
                time.sleep(min(delay, 10))
        except Exception as e:
            if retry < max_retries - 1:
                delay = initial_delay * (2 ** min(retry, 5))
                time.sleep(min(delay, 10))
    
    return None


def find_failed_runs(dataset_dir: str = "../dataset"):
    """
    扫描数据集，找出所有失败的 run
    
    Returns:
        List of (json_file_path, run_index, image_path)
    """
    script_dir = Path(__file__).parent
    dataset_path = script_dir / dataset_dir
    
    failed_runs = []
    
    char_folders = sorted([
        d for d in dataset_path.iterdir()
        if d.is_dir() and len(d.name) > 0 and '\u4e00' <= d.name[0] <= '\u9fff'
    ])
    
    for char_folder in char_folders:
        recognition_dir = char_folder / "recognition"
        if not recognition_dir.exists():
            continue
        
        for json_file in recognition_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            image_path = dataset_path / data['image_path']
            
            for run in data['runs']:
                if run['llm_output'] is None:
                    failed_runs.append({
                        'json_file': json_file,
                        'run_id': run['run_id'],
                        'image_path': image_path,
                        'character': data['character'],
                        'stroke_name': data['stroke_name']
                    })
    
    return failed_runs


def refill_failed_data(dataset_dir: str = "../dataset", dry_run: bool = False):
    """
    补录所有失败的数据
    """
    print("\n扫描失败的识别...")
    failed_runs = find_failed_runs(dataset_dir)
    
    if len(failed_runs) == 0:
        print("✅ 没有发现失败的识别！")
        return
    
    print(f"发现 {len(failed_runs)} 个失败的识别")
    
    if dry_run:
        print("\n【Dry-run 模式】只显示，不执行补录")
        for item in failed_runs[:10]:
            print(f"  - {item['character']}/{item['stroke_name']}, run {item['run_id']}")
        if len(failed_runs) > 10:
            print(f"  ... 还有 {len(failed_runs) - 10} 个")
        return
    
    print("\n开始补录...")
    success_count = 0
    
    with tqdm(total=len(failed_runs), desc="补录进度", unit="次") as pbar:
        for item in failed_runs:
            # 重新识别
            result = recognize_image_once(item['image_path'])
            
            if result is not None:
                # 读取原 JSON
                with open(item['json_file'], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 更新对应的 run
                for run in data['runs']:
                    if run['run_id'] == item['run_id']:
                        run['llm_output'] = result
                        break
                
                # 保存更新后的 JSON
                with open(item['json_file'], 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                success_count += 1
            
            pbar.update(1)
            pbar.set_postfix_str(f"成功: {success_count}/{len(failed_runs)}")
    
    print(f"\n✅ 补录完成！")
    print(f"   成功: {success_count}/{len(failed_runs)}")
    print(f"   失败: {len(failed_runs) - success_count}/{len(failed_runs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="补录失败的识别数据")
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="../dataset",
        help="数据集目录 (默认: ../dataset)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计，不执行补录"
    )
    
    args = parser.parse_args()
    
    refill_failed_data(dataset_dir=args.dataset, dry_run=args.dry_run)
