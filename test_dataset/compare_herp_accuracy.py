#!/usr/bin/env python3
"""
比较 baseline_1 (recognition_1) 和 Chain-of-Frames (COF_input) 
在 HERP 点的 top1 概率差

统计正确答案与概率 top1 的概率差的平均值
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import statistics


def get_probability_gap(llm_output: Optional[Dict], correct_char: str) -> Optional[float]:
    """
    计算 top1 概率和正确答案概率的差值
    
    返回: top1_prob - correct_answer_prob
    
    如果 top1 就是正确答案，差值为 0
    如果正确答案不在 top3，返回 None
    """
    if llm_output is None:
        return None
    
    probabilities = llm_output.get('probabilities', [])
    if not probabilities:
        return None
    
    top1 = probabilities[0]
    top1_prob = top1.get('p', 0.0)
    
    # 查找正确答案的概率
    correct_prob = None
    for item in probabilities:
        if item.get('char') == correct_char:
            correct_prob = item.get('p', 0.0)
            break
    
    # 如果正确答案不在 top3 中，返回 None
    if correct_prob is None:
        return None
    
    # 返回差值：top1 - 正确答案
    return top1_prob - correct_prob


def analyze_character(char_folder: Path) -> Optional[Dict]:
    """
    分析单个字符的 HERP 点识别结果
    
    返回:
        {
            'character': '字',
            'herp': 9,
            'baseline_gaps': [0.0, 0.0, ...],  # 5次运行的概率差值 (top1 - 正确答案)
            'cof_gaps': [0.0, 0.1, ...],
            'baseline_avg_gap': 0.02,  # 平均概率差
            'cof_avg_gap': 0.05,
            'baseline_valid_count': 5,  # 正确答案在 top3 中的次数
            'cof_valid_count': 5,
            'gap_diff': -0.03  # baseline - cof（负数表示baseline差值更小，即更准确）
        }
    """
    character = char_folder.name
    
    # 读取 stroke_info.json 获取 HERP
    stroke_info_file = char_folder / "stroke_info.json"
    if not stroke_info_file.exists():
        print(f"  ⚠️ {character}: 找不到 stroke_info.json")
        return None
    
    with open(stroke_info_file, 'r', encoding='utf-8') as f:
        stroke_info = json.load(f)
    
    herp = stroke_info.get('HERP')
    if herp is None:
        print(f"  ⚠️ {character}: stroke_info.json 中没有 HERP")
        return None
    
    # 读取 baseline_1 的结果 (recognition_1/stroke_{HERP}.json)
    baseline_file = char_folder / "recognition_1" / f"stroke_{herp:02d}.json"
    if not baseline_file.exists():
        print(f"  ⚠️ {character}: 找不到 baseline_1 结果 {baseline_file.name}")
        return None
    
    with open(baseline_file, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)
    
    # 读取 COF 的结果 (COF_input/recognition_result.json)
    cof_file = char_folder / "COF_input" / "recognition_result.json"
    if not cof_file.exists():
        print(f"  ⚠️ {character}: 找不到 COF 结果")
        return None
    
    with open(cof_file, 'r', encoding='utf-8') as f:
        cof_data = json.load(f)
    
    # 提取 baseline 的 5 次运行结果
    baseline_gaps = []
    baseline_valid_count = 0
    for run in baseline_data.get('runs', []):
        llm_output = run.get('llm_output')
        gap = get_probability_gap(llm_output, character)
        if gap is not None:
            baseline_gaps.append(gap)
            baseline_valid_count += 1
    
    # 提取 COF 的 5 次运行结果
    cof_gaps = []
    cof_valid_count = 0
    for run in cof_data.get('runs', []):
        llm_output = run.get('llm_output')
        gap = get_probability_gap(llm_output, character)
        if gap is not None:
            cof_gaps.append(gap)
            cof_valid_count += 1
    
    # 计算平均概率差（只计算正确答案在 top3 中的）
    baseline_avg_gap = statistics.mean(baseline_gaps) if baseline_gaps else None
    cof_avg_gap = statistics.mean(cof_gaps) if cof_gaps else None
    
    # 计算差值的差值（baseline - cof）
    # 负数表示 baseline 的差值更小（即更准确）
    if baseline_avg_gap is not None and cof_avg_gap is not None:
        gap_diff = baseline_avg_gap - cof_avg_gap
    else:
        gap_diff = None
    
    return {
        'character': character,
        'herp': herp,
        'baseline_gaps': baseline_gaps,
        'cof_gaps': cof_gaps,
        'baseline_avg_gap': round(baseline_avg_gap, 4) if baseline_avg_gap is not None else None,
        'cof_avg_gap': round(cof_avg_gap, 4) if cof_avg_gap is not None else None,
        'baseline_valid_count': baseline_valid_count,
        'cof_valid_count': cof_valid_count,
        'gap_diff': round(gap_diff, 4) if gap_diff is not None else None
    }


def main():
    """主函数"""
    dataset_path = Path(__file__).parent
    
    # 获取所有汉字文件夹
    char_folders = sorted([
        d for d in dataset_path.iterdir()
        if d.is_dir() and len(d.name) > 0 and '\u4e00' <= d.name[0] <= '\u9fff'
    ])
    
    print(f"\n{'='*80}")
    print(f"🎯 比较 baseline_1 和 Chain-of-Frames 在 HERP 点的概率差值")
    print(f"   (概率差 = top1概率 - 正确答案概率，越小越好)")
    print(f"{'='*80}")
    print(f"数据集: {dataset_path}")
    print(f"字符数: {len(char_folders)}\n")
    
    results = []
    
    for char_folder in char_folders:
        result = analyze_character(char_folder)
        if result:
            results.append(result)
            baseline_gap = result['baseline_avg_gap']
            cof_gap = result['cof_avg_gap']
            gap_diff = result['gap_diff']
            
            baseline_str = f"{baseline_gap:.4f}" if baseline_gap is not None else "N/A"
            cof_str = f"{cof_gap:.4f}" if cof_gap is not None else "N/A"
            gap_diff_str = f"{gap_diff:+.4f}" if gap_diff is not None else "N/A"
            
            print(f"  ✅ {result['character']}: "
                  f"baseline={baseline_str} ({result['baseline_valid_count']}/5), "
                  f"COF={cof_str} ({result['cof_valid_count']}/5), "
                  f"差值差={gap_diff_str}")
    
    # 计算总体统计
    if results:
        # 只统计有效的结果
        valid_baseline_gaps = [r['baseline_avg_gap'] for r in results if r['baseline_avg_gap'] is not None]
        valid_cof_gaps = [r['cof_avg_gap'] for r in results if r['cof_avg_gap'] is not None]
        valid_gap_diffs = [r['gap_diff'] for r in results if r['gap_diff'] is not None]
        
        overall_baseline_avg_gap = statistics.mean(valid_baseline_gaps) if valid_baseline_gaps else None
        overall_cof_avg_gap = statistics.mean(valid_cof_gaps) if valid_cof_gaps else None
        overall_gap_diff = statistics.mean(valid_gap_diffs) if valid_gap_diffs else None
        
        # 计算正确答案在 top3 中的总次数
        baseline_total_valid = sum(r['baseline_valid_count'] for r in results)
        cof_total_valid = sum(r['cof_valid_count'] for r in results)
        total_runs = len(results) * 5
        
        baseline_top3_rate = baseline_total_valid / total_runs * 100
        cof_top3_rate = cof_total_valid / total_runs * 100
        
        # 计算 top1 准确率（gap = 0 的次数）
        baseline_top1_correct = sum(1 for r in results for gap in r['baseline_gaps'] if gap == 0.0)
        cof_top1_correct = sum(1 for r in results for gap in r['cof_gaps'] if gap == 0.0)
        
        baseline_top1_rate = baseline_top1_correct / total_runs * 100
        cof_top1_rate = cof_top1_correct / total_runs * 100
        
        summary = {
            'dataset': 'test_dataset',
            'total_characters': len(results),
            'total_runs_per_method': total_runs,
            'baseline_1': {
                'method': 'Single frame at HERP',
                'avg_probability_gap': round(overall_baseline_avg_gap, 4) if overall_baseline_avg_gap is not None else None,
                'description': 'top1概率 - 正确答案概率',
                'correct_in_top3': baseline_total_valid,
                'top3_rate': round(baseline_top3_rate, 2),
                'top1_correct': baseline_top1_correct,
                'top1_accuracy': round(baseline_top1_rate, 2)
            },
            'chain_of_frames': {
                'method': 'Multi-frame (4 frames + HERP)',
                'avg_probability_gap': round(overall_cof_avg_gap, 4) if overall_cof_avg_gap is not None else None,
                'description': 'top1概率 - 正确答案概率',
                'correct_in_top3': cof_total_valid,
                'top3_rate': round(cof_top3_rate, 2),
                'top1_correct': cof_top1_correct,
                'top1_accuracy': round(cof_top1_rate, 2)
            },
            'comparison': {
                'avg_gap_diff': round(overall_gap_diff, 4) if overall_gap_diff is not None else None,
                'description': 'baseline - COF (负数表示baseline的gap更小，即更准确)',
                'top3_rate_diff': round(baseline_top3_rate - cof_top3_rate, 2),
                'top1_accuracy_diff': round(baseline_top1_rate - cof_top1_rate, 2),
                'winner': 'baseline_1' if overall_gap_diff and overall_gap_diff < 0 else 'chain_of_frames'
            },
            'per_character_results': results
        }
        
        # 保存到 JSON
        output_file = dataset_path / "herp_comparison_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*80}")
        print(f"📊 总体统计")
        print(f"{'='*80}")
        print(f"Baseline_1 (单帧):")
        if overall_baseline_avg_gap is not None:
            print(f"  - 平均概率差: {overall_baseline_avg_gap:.4f}")
        print(f"  - 正确答案在 top3: {baseline_total_valid}/{total_runs} ({baseline_top3_rate:.1f}%)")
        print(f"  - top1 准确率: {baseline_top1_correct}/{total_runs} ({baseline_top1_rate:.1f}%)")
        
        print(f"\nChain-of-Frames (多帧):")
        if overall_cof_avg_gap is not None:
            print(f"  - 平均概率差: {overall_cof_avg_gap:.4f}")
        print(f"  - 正确答案在 top3: {cof_total_valid}/{total_runs} ({cof_top3_rate:.1f}%)")
        print(f"  - top1 准确率: {cof_top1_correct}/{total_runs} ({cof_top1_rate:.1f}%)")
        
        if overall_gap_diff is not None:
            print(f"\n概率差的差值 (baseline - COF): {overall_gap_diff:+.4f}")
            if overall_gap_diff < 0:
                print(f"  → Baseline 的概率差更小，表现更好")
            else:
                print(f"  → COF 的概率差更小，表现更好")
        
        print(f"\n{'='*80}")
        print(f"✅ 结果已保存到: {output_file}")
        print(f"{'='*80}\n")
    else:
        print("\n❌ 没有找到有效的数据\n")


if __name__ == "__main__":
    main()
