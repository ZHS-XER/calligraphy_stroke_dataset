#!/usr/bin/env python3
"""
统计当前 dataset 的处理进度、速度和失败率
"""

import json
from pathlib import Path
from datetime import datetime
import os

def count_detailed_stats(dataset_dir="../dataset"):
    """详细统计"""
    total_runs = 0
    failed_runs = 0
    total_chars = 0
    processed_chars = 0
    total_strokes = 0
    processed_strokes = 0
    
    # 统计每个字符的处理时间（通过文件修改时间）
    first_time = None
    last_time = None
    
    dataset = Path(dataset_dir)
    
    # 获取所有汉字文件夹
    char_folders = sorted([
        d for d in dataset.iterdir()
        if d.is_dir() and len(d.name) > 0 and '\u4e00' <= d.name[0] <= '\u9fff'
    ])
    
    total_chars = len(char_folders)
    
    for char_dir in char_folders:
        rec_dir = char_dir / "recognition"
        
        # 统计该字的笔画数
        stroke_files = sorted([
            f for f in char_dir.glob("stroke_*.png")
            if f.name != "stroke_00.png"
        ])
        
        if not rec_dir.exists():
            continue
        
        json_files = list(rec_dir.glob("*.json"))
        
        if len(json_files) == 0:
            continue
        
        # 这个字符至少开始处理了
        char_has_data = False
        char_stroke_count = 0
        
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                runs = data.get('runs', [])
                if len(runs) > 0:
                    char_has_data = True
                    char_stroke_count += 1
                    
                for run in runs:
                    total_runs += 1
                    if run['llm_output'] is None:
                        failed_runs += 1
                
                # 记录文件时间
                mtime = json_file.stat().st_mtime
                file_time = datetime.fromtimestamp(mtime)
                
                if first_time is None or file_time < first_time:
                    first_time = file_time
                if last_time is None or file_time > last_time:
                    last_time = file_time
                    
            except Exception as e:
                print(f"⚠️ 读取文件出错: {json_file}: {e}")
                pass
        
        if char_has_data:
            processed_chars += 1
            processed_strokes += char_stroke_count
            total_strokes += len(stroke_files)
    
    return {
        'total_chars': total_chars,
        'processed_chars': processed_chars,
        'total_strokes': total_strokes,
        'processed_strokes': processed_strokes,
        'total_runs': total_runs,
        'failed_runs': failed_runs,
        'first_time': first_time,
        'last_time': last_time
    }


def main():
    print("\n" + "=" * 70)
    print("📊 数据集处理统计")
    print("=" * 70)
    
    stats = count_detailed_stats()
    
    # 基本统计
    print(f"\n📁 处理进度:")
    print(f"  总字符数: {stats['total_chars']}")
    print(f"  已处理字符: {stats['processed_chars']} ({stats['processed_chars']/stats['total_chars']*100:.1f}%)")
    print(f"  剩余字符: {stats['total_chars'] - stats['processed_chars']}")
    
    if stats['processed_strokes'] > 0:
        print(f"\n📝 笔画统计:")
        print(f"  已处理笔画: {stats['processed_strokes']}")
        print(f"  平均笔画/字: {stats['processed_strokes']/stats['processed_chars']:.1f}")
    
    # API 调用统计
    print(f"\n🔄 API 调用:")
    print(f"  总调用次数: {stats['total_runs']}")
    print(f"  成功次数: {stats['total_runs'] - stats['failed_runs']}")
    print(f"  失败次数: {stats['failed_runs']}")
    print(f"  失败率: {stats['failed_runs']/stats['total_runs']*100:.2f}%" if stats['total_runs'] > 0 else "  失败率: N/A")
    
    # 时间统计
    if stats['first_time'] and stats['last_time']:
        elapsed = stats['last_time'] - stats['first_time']
        elapsed_hours = elapsed.total_seconds() / 3600
        elapsed_days = elapsed_hours / 24
        
        print(f"\n⏱️  时间统计:")
        print(f"  开始时间: {stats['first_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  最后更新: {stats['last_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  总耗时: {elapsed_days:.2f} 天 ({elapsed_hours:.1f} 小时)")
        
        # 速度统计
        if stats['processed_chars'] > 0 and elapsed_hours > 0:
            chars_per_hour = stats['processed_chars'] / elapsed_hours
            hours_per_char = elapsed_hours / stats['processed_chars']
            
            print(f"\n⚡ 处理速度:")
            print(f"  字符/小时: {chars_per_hour:.2f}")
            print(f"  小时/字符: {hours_per_char:.2f}")
            
            if stats['total_runs'] > 0:
                calls_per_hour = stats['total_runs'] / elapsed_hours
                seconds_per_call = elapsed.total_seconds() / stats['total_runs']
                
                print(f"  API调用/小时: {calls_per_hour:.1f}")
                print(f"  秒/API调用: {seconds_per_call:.2f}")
        
        # 预估剩余时间
        if stats['processed_chars'] > 0:
            remaining_chars = stats['total_chars'] - stats['processed_chars']
            estimated_hours = remaining_chars * hours_per_char
            estimated_days = estimated_hours / 24
            
            print(f"\n🔮 预估剩余:")
            print(f"  剩余字符: {remaining_chars}")
            print(f"  预估时间: {estimated_days:.2f} 天 ({estimated_hours:.1f} 小时)")
            
            # 预估完成时间
            from datetime import timedelta
            estimated_finish = stats['last_time'] + timedelta(hours=estimated_hours)
            print(f"  预计完成: {estimated_finish.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "=" * 70 + "\n")
    
    # 保存统计结果
    output = {
        'timestamp': datetime.now().isoformat(),
        'total_chars': stats['total_chars'],
        'processed_chars': stats['processed_chars'],
        'progress_percent': stats['processed_chars']/stats['total_chars']*100 if stats['total_chars'] > 0 else 0,
        'total_runs': stats['total_runs'],
        'failed_runs': stats['failed_runs'],
        'failure_rate': stats['failed_runs']/stats['total_runs']*100 if stats['total_runs'] > 0 else 0,
        'elapsed_hours': elapsed_hours if stats['first_time'] and stats['last_time'] else 0,
        'chars_per_hour': chars_per_hour if stats['processed_chars'] > 0 and elapsed_hours > 0 else 0,
        'seconds_per_call': seconds_per_call if stats['total_runs'] > 0 and elapsed_hours > 0 else 0
    }
    
    with open('current_stats.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print("💾 详细统计已保存到: current_stats.json\n")


if __name__ == "__main__":
    main()
