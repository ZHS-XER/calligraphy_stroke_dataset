#!/usr/bin/env python3
"""
验证 COF 输入增强版输出的脚本

检查：
1. 每个字符的倒数三笔是否都有输出
2. 每个 JSON 文件是否完整（5 次运行）
3. 每个 llm_output 是否有效
4. 重试统计是否正确
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def validate_llm_output(llm_output: Dict) -> bool:
    """验证 LLM 输出是否有效"""
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


def validate_stroke_file(file_path: Path) -> Tuple[bool, str]:
    """
    验证单个笔画文件
    
    Returns:
        (是否有效, 错误信息)
    """
    if not file_path.exists():
        return False, "文件不存在"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return False, f"JSON 解析失败: {str(e)}"
    
    # 检查必须字段
    required_fields = ['character', 'stroke_name', 'target_stroke', 
                      'remaining_strokes', 'total_strokes', 'input_frames', 'runs']
    for field in required_fields:
        if field not in data:
            return False, f"缺少字段: {field}"
    
    # 检查 input_frames 数量
    if len(data['input_frames']) != 5:
        return False, f"input_frames 数量错误: {len(data['input_frames'])} (应为 5)"
    
    # 检查 runs 数量
    if len(data['runs']) != 5:
        return False, f"runs 数量错误: {len(data['runs'])} (应为 5)"
    
    # 检查每个 run
    for i, run in enumerate(data['runs'], 1):
        if 'run_id' not in run or run['run_id'] != i:
            return False, f"run_id 错误: run {i}"
        
        if 'llm_output' not in run:
            return False, f"缺少 llm_output: run {i}"
        
        if not validate_llm_output(run['llm_output']):
            return False, f"llm_output 无效: run {i}"
        
        if 'retry_count' not in run:
            return False, f"缺少 retry_count: run {i}"
    
    return True, "OK"


def validate_character(char_folder: Path) -> Tuple[bool, List[str]]:
    """
    验证单个字符的输出
    
    Returns:
        (是否完整, 错误/警告列表)
    """
    messages = []
    
    # 读取 stroke_info.json
    stroke_info_file = char_folder / "stroke_info.json"
    if not stroke_info_file.exists():
        messages.append("⚠️ 缺少 stroke_info.json")
        return False, messages
    
    try:
        with open(stroke_info_file, 'r', encoding='utf-8') as f:
            stroke_info = json.load(f)
        stroke_count = stroke_info.get('stroke_count', len(stroke_info.get('strokes', [])))
    except Exception as e:
        messages.append(f"⚠️ 读取 stroke_info.json 失败: {str(e)}")
        return False, messages
    
    # 检查 COF_input 文件夹
    cof_dir = char_folder / "COF_input"
    if not cof_dir.exists():
        messages.append("⚠️ 缺少 COF_input 文件夹")
        return False, messages
    
    # 检查倒数三笔
    all_valid = True
    for offset in [1, 2, 3]:
        target_idx = stroke_count - offset + 1
        if target_idx < 1:
            continue
        
        stroke_file = cof_dir / f"stroke_{target_idx:02d}.json"
        valid, error = validate_stroke_file(stroke_file)
        
        if valid:
            messages.append(f"✅ stroke_{target_idx:02d}.json: {error}")
        else:
            messages.append(f"❌ stroke_{target_idx:02d}.json: {error}")
            all_valid = False
    
    return all_valid, messages


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python validate_output.py <数据集目录>")
        print("示例: python validate_output.py ../test_dataset")
        sys.exit(1)
    
    dataset_dir = Path(sys.argv[1])
    if not dataset_dir.exists():
        print(f"❌ 数据集目录不存在: {dataset_dir}")
        sys.exit(1)
    
    # 获取所有汉字文件夹
    char_folders = sorted([
        d for d in dataset_dir.iterdir()
        if d.is_dir() and len(d.name) > 0 and '\u4e00' <= d.name[0] <= '\u9fff'
    ])
    
    print(f"\n{'='*60}")
    print(f"🔍 验证 COF 输入增强版输出")
    print(f"{'='*60}")
    print(f"数据集: {dataset_dir}")
    print(f"字符数: {len(char_folders)}")
    print(f"{'='*60}\n")
    
    valid_count = 0
    invalid_count = 0
    
    for char_folder in char_folders:
        character = char_folder.name
        valid, messages = validate_character(char_folder)
        
        print(f"\n字符: '{character}'")
        for msg in messages:
            print(f"  {msg}")
        
        if valid:
            valid_count += 1
        else:
            invalid_count += 1
    
    print(f"\n{'='*60}")
    print(f"📊 验证结果")
    print(f"{'='*60}")
    print(f"✅ 有效: {valid_count}")
    print(f"❌ 无效: {invalid_count}")
    print(f"总计: {len(char_folders)}")
    print(f"{'='*60}\n")
    
    if invalid_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
