#!/usr/bin/env python3
"""
测试改进后的 resume 机制
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from recognize_strokes import is_stroke_file_valid, get_stroke_images

def test_incomplete_characters():
    """测试不完整字符的检测"""
    print("\n" + "="*60)
    print("测试 resume 机制对不完整文件的检测")
    print("="*60)
    
    dataset_path = Path(__file__).parent.parent / "dataset"
    
    # 测试已知有问题的字符
    test_chars = ["套", "奠", "妆"]
    
    for char_name in test_chars:
        char_folder = dataset_path / char_name
        if not char_folder.exists():
            print(f"\n⚠️  字符 '{char_name}' 不存在，跳过")
            continue
        
        recognition_dir = char_folder / "recognition"
        if not recognition_dir.exists():
            print(f"\n⚠️  字符 '{char_name}' 没有 recognition 文件夹，跳过")
            continue
        
        print(f"\n检查字符: '{char_name}'")
        
        # 获取预期的笔画文件
        stroke_images = get_stroke_images(char_folder)
        expected_files = {f"{img.stem}.json": img for img in stroke_images}
        
        print(f"  预期文件数: {len(expected_files)}")
        
        # 检查每个文件
        all_valid = True
        invalid_files = []
        
        for json_name, img_path in expected_files.items():
            json_path = recognition_dir / json_name
            is_valid = is_stroke_file_valid(json_path)
            
            if not is_valid:
                all_valid = False
                invalid_files.append(json_name)
                
                # 读取文件查看详情
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    print(f"  ❌ {json_name}: 无效")
                    
                    # 检查哪些 run 有问题
                    if 'runs' in data:
                        for i, run in enumerate(data['runs'], 1):
                            llm_output = run.get('llm_output')
                            if llm_output is None:
                                print(f"      - Run {i}: llm_output 为 None")
                            elif not isinstance(llm_output, dict):
                                print(f"      - Run {i}: llm_output 类型错误")
                            elif 'answer' not in llm_output or not llm_output.get('answer'):
                                print(f"      - Run {i}: answer 缺失或为空")
                except Exception as e:
                    print(f"  ❌ {json_name}: 无法读取 ({str(e)[:50]})")
            else:
                print(f"  ✅ {json_name}: 有效")
        
        # 总结
        if all_valid:
            print(f"  ✅ 字符 '{char_name}' 所有文件都有效（不应该出现在不完整列表中）")
        else:
            print(f"  ❌ 字符 '{char_name}' 有 {len(invalid_files)} 个无效文件")
            print(f"  👉 Resume 应该重新处理这个字符")


def test_complete_characters():
    """测试完整字符的检测"""
    print("\n" + "="*60)
    print("测试完整字符的检测")
    print("="*60)
    
    dataset_path = Path(__file__).parent.parent / "dataset"
    
    # 随机找几个字符测试
    count = 0
    for char_folder in dataset_path.iterdir():
        if not char_folder.is_dir():
            continue
        
        char_name = char_folder.name
        if not ('\u4e00' <= char_name[0] <= '\u9fff'):
            continue
        
        recognition_dir = char_folder / "recognition"
        if not recognition_dir.exists():
            continue
        
        # 获取预期的笔画文件
        stroke_images = get_stroke_images(char_folder)
        if not stroke_images:
            continue
        
        expected_files = {f"{img.stem}.json": img for img in stroke_images}
        
        # 检查所有文件是否有效
        all_valid = True
        for json_name, img_path in expected_files.items():
            json_path = recognition_dir / json_name
            if not is_stroke_file_valid(json_path):
                all_valid = False
                break
        
        if all_valid and len(expected_files) > 0:
            print(f"✅ 字符 '{char_name}': 所有 {len(expected_files)} 个文件都有效")
            count += 1
            
            if count >= 5:
                break
    
    if count == 0:
        print("⚠️  未找到完整的字符")
    else:
        print(f"\n✅ 成功验证 {count} 个完整字符")


if __name__ == "__main__":
    try:
        test_incomplete_characters()
        test_complete_characters()
        print("\n" + "="*60)
        print("🎉 Resume 机制测试完成！")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n❌ 测试错误: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
