#!/usr/bin/env python3
"""
检查 baseline 运行产生的 recognition 文件夹中的 JSON 文件是否完整
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def check_json_file(json_file: Path) -> Tuple[bool, List[str]]:
    """
    检查单个 JSON 文件是否完整
    
    Returns:
        (is_complete, errors): 是否完整，错误列表
    """
    errors = []
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查必需的顶层字段
        required_fields = ['character', 'stroke_name', 'image_path', 'runs']
        for field in required_fields:
            if field not in data:
                errors.append(f"缺少字段: {field}")
        
        # 检查 runs 数组
        if 'runs' in data:
            if not isinstance(data['runs'], list):
                errors.append("runs 不是数组")
            elif len(data['runs']) != 5:
                errors.append(f"runs 数量错误: {len(data['runs'])} (预期 5)")
            else:
                # 检查每个 run
                for i, run in enumerate(data['runs'], 1):
                    if not isinstance(run, dict):
                        errors.append(f"run {i} 不是对象")
                        continue
                    
                    if 'run_id' not in run:
                        errors.append(f"run {i} 缺少 run_id")
                    elif run['run_id'] != i:
                        errors.append(f"run {i} 的 run_id 错误: {run['run_id']}")
                    
                    if 'llm_output' not in run:
                        errors.append(f"run {i} 缺少 llm_output")
                    elif run['llm_output'] is None:
                        errors.append(f"run {i} 的 llm_output 为 None (识别失败)")
                    elif not isinstance(run['llm_output'], dict):
                        errors.append(f"run {i} 的 llm_output 不是对象")
                    else:
                        # 检查 llm_output 的结构
                        llm_output = run['llm_output']
                        if 'answer' not in llm_output:
                            errors.append(f"run {i} 的 llm_output 缺少 answer")
                        if 'probabilities' not in llm_output:
                            errors.append(f"run {i} 的 llm_output 缺少 probabilities")
                        if 'rationale' not in llm_output:
                            errors.append(f"run {i} 的 llm_output 缺少 rationale")
        
    except json.JSONDecodeError as e:
        errors.append(f"JSON 解析失败: {str(e)}")
    except Exception as e:
        errors.append(f"读取文件失败: {str(e)}")
    
    return (len(errors) == 0, errors)


def check_character_folder(char_folder: Path) -> Dict:
    """
    检查单个字符文件夹的 recognition 目录
    
    Returns:
        检查结果字典
    """
    recognition_dir = char_folder / "recognition"
    
    if not recognition_dir.exists():
        return {
            'character': char_folder.name,
            'status': 'missing',
            'error': 'recognition 文件夹不存在'
        }
    
    # 获取预期的笔画文件
    stroke_files = sorted([
        f for f in char_folder.glob("stroke_*.png")
        if f.name != "stroke_00.png"
    ])
    expected_strokes = stroke_files[-5:] if len(stroke_files) > 5 else stroke_files
    expected_json_files = {f"{stroke.stem}.json" for stroke in expected_strokes}
    
    # 获取实际的 JSON 文件
    actual_json_files = {f.name for f in recognition_dir.glob("*.json")}
    
    # 检查缺失的文件
    missing_files = expected_json_files - actual_json_files
    extra_files = actual_json_files - expected_json_files
    
    # 检查每个 JSON 文件的完整性
    file_errors = {}
    for json_filename in expected_json_files:
        if json_filename in actual_json_files:
            json_path = recognition_dir / json_filename
            is_complete, errors = check_json_file(json_path)
            if not is_complete:
                file_errors[json_filename] = errors
    
    # 判断整体状态
    if missing_files or file_errors:
        status = 'incomplete'
    elif extra_files:
        status = 'complete_with_extra'
    else:
        status = 'complete'
    
    result = {
        'character': char_folder.name,
        'status': status,
        'expected_files': len(expected_json_files),
        'actual_files': len(actual_json_files),
    }
    
    if missing_files:
        result['missing_files'] = sorted(missing_files)
    if extra_files:
        result['extra_files'] = sorted(extra_files)
    if file_errors:
        result['file_errors'] = file_errors
    
    return result


def check_dataset(dataset_dir: str = "../dataset") -> Dict:
    """
    检查整个数据集的 recognition 文件夹
    
    Returns:
        检查报告
    """
    script_dir = Path(__file__).parent
    dataset_path = script_dir / dataset_dir
    
    if not dataset_path.exists():
        print(f"❌ 错误: 数据集目录不存在 - {dataset_path}")
        return {}
    
    # 获取所有汉字文件夹
    char_folders = sorted([
        d for d in dataset_path.iterdir()
        if d.is_dir() and len(d.name) > 0 and '\u4e00' <= d.name[0] <= '\u9fff'
    ])
    
    print(f"\n{'='*60}")
    print(f"🔍 检查 recognition 文件夹完整性")
    print(f"{'='*60}")
    print(f"数据集: {dataset_path}")
    print(f"字符数: {len(char_folders)}")
    print(f"{'='*60}\n")
    
    # 统计计数器
    stats = {
        'total': len(char_folders),
        'complete': 0,
        'incomplete': 0,
        'missing': 0,
        'complete_with_extra': 0
    }
    
    # 存储有问题的字符
    problematic_chars = {
        'incomplete': [],
        'missing': [],
    }
    
    # 检查每个字符
    for char_folder in char_folders:
        result = check_character_folder(char_folder)
        status = result['status']
        stats[status] += 1
        
        if status == 'incomplete':
            problematic_chars['incomplete'].append(result)
        elif status == 'missing':
            problematic_chars['missing'].append(result)
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"📊 统计结果")
    print(f"{'='*60}")
    print(f"总字符数: {stats['total']}")
    print(f"✅ 完整: {stats['complete']}")
    print(f"⚠️  不完整: {stats['incomplete']}")
    print(f"❌ 缺失: {stats['missing']}")
    if stats['complete_with_extra'] > 0:
        print(f"ℹ️  有额外文件: {stats['complete_with_extra']}")
    print(f"{'='*60}\n")
    
    # 显示有问题的字符
    if problematic_chars['missing']:
        print(f"\n{'='*60}")
        print(f"❌ 缺失 recognition 文件夹的字符 ({len(problematic_chars['missing'])})")
        print(f"{'='*60}")
        for result in problematic_chars['missing'][:20]:  # 最多显示20个
            print(f"  {result['character']}: {result['error']}")
        if len(problematic_chars['missing']) > 20:
            print(f"  ... 还有 {len(problematic_chars['missing']) - 20} 个")
        print()
    
    if problematic_chars['incomplete']:
        print(f"\n{'='*60}")
        print(f"⚠️  不完整的字符 ({len(problematic_chars['incomplete'])})")
        print(f"{'='*60}")
        for result in problematic_chars['incomplete'][:20]:  # 最多显示20个
            print(f"\n  字符: {result['character']}")
            if 'missing_files' in result:
                print(f"    缺失文件: {', '.join(result['missing_files'])}")
            if 'file_errors' in result:
                print(f"    有错误的文件:")
                for filename, errors in result['file_errors'].items():
                    print(f"      {filename}:")
                    for error in errors:
                        print(f"        - {error}")
        if len(problematic_chars['incomplete']) > 20:
            print(f"\n  ... 还有 {len(problematic_chars['incomplete']) - 20} 个")
        print()
    
    # 保存完整报告
    report_file = script_dir / "completeness_report.json"
    report = {
        'stats': stats,
        'problematic_chars': problematic_chars
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 完整报告已保存到: {report_file}\n")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="检查 recognition 文件夹的完整性"
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="../dataset",
        help="数据集目录 (默认: ../dataset)"
    )
    
    args = parser.parse_args()
    check_dataset(dataset_dir=args.dataset)
