#!/usr/bin/env python3
"""
测试完整的 resume 流程（只检测，不实际运行 API）
"""

import sys
from pathlib import Path

# 模拟运行 recognize_strokes.py 的 resume 检查部分
sys.path.insert(0, str(Path(__file__).parent))

from recognize_strokes import get_stroke_images, is_stroke_file_valid

def simulate_resume_check():
    """模拟 process_dataset 的 resume 检查逻辑"""
    script_dir = Path(__file__).parent
    dataset_path = script_dir.parent / "dataset"
    
    if not dataset_path.exists():
        print(f"❌ 错误: 数据集目录不存在 - {dataset_path}")
        return
    
    # 获取所有汉字文件夹
    char_folders = sorted([
        d for d in dataset_path.iterdir()
        if d.is_dir() and len(d.name) > 0 and '\u4e00' <= d.name[0] <= '\u9fff'
    ])
    
    print(f"\n{'='*60}")
    print(f"🔍 模拟 Resume 检查")
    print(f"{'='*60}")
    print(f"数据集: {dataset_path}")
    print(f"字符数: {len(char_folders)}")
    print(f"{'='*60}\n")
    
    # 断点续传：检查哪些字符已经处理完成
    processed_chars = set()
    incomplete_chars = []  # 记录需要重新处理的字符
    
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
    
    print(f"\n待处理: {len(remaining_folders)}")
    print(f"{'='*60}\n")
    
    # 详细统计
    print("📊 详细统计:")
    print(f"  总字符数: {len(char_folders)}")
    print(f"  已完成（完整）: {len(processed_chars)}")
    print(f"  不完整（需重新处理）: {len(incomplete_chars)}")
    print(f"  未处理: {len(remaining_folders) - len(incomplete_chars)}")
    print(f"  待处理总计: {len(remaining_folders)}")
    
    # 验证几个已知不完整的字符
    print(f"\n{'='*60}")
    print("✅ 验证已知不完整字符:")
    print(f"{'='*60}")
    
    known_incomplete = ["套", "奠", "妆"]
    for char in known_incomplete:
        if char in incomplete_chars:
            print(f"  ✅ '{char}' - 正确识别为不完整")
        elif char in processed_chars:
            print(f"  ❌ '{char}' - 错误标记为已完成")
        else:
            print(f"  ⚠️  '{char}' - 未找到")
    
    print(f"\n{'='*60}")
    print("🎉 Resume 检查完成！")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        simulate_resume_check()
    except Exception as e:
        print(f"\n❌ 错误: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
