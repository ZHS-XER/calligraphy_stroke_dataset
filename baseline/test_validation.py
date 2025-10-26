#!/usr/bin/env python3
"""
测试新增的验证函数
"""

import json
from pathlib import Path
import sys

# 导入需要测试的函数
sys.path.insert(0, str(Path(__file__).parent))
from recognize_strokes import validate_llm_output, is_stroke_file_valid

def test_validate_llm_output():
    """测试 validate_llm_output 函数"""
    print("\n" + "="*60)
    print("测试 validate_llm_output() 函数")
    print("="*60)
    
    # 测试用例1: 有效的输出
    valid_output = {
        "answer": "爱",
        "probabilities": [
            {"char": "爱", "p": 0.80},
            {"char": "受", "p": 0.15},
            {"char": "愛", "p": 0.05}
        ],
        "rationale": "根据笔画判断"
    }
    result = validate_llm_output(valid_output)
    print(f"✅ 有效输出: {result} (预期: True)")
    assert result == True
    
    # 测试用例2: None 输出
    result = validate_llm_output(None)
    print(f"✅ None 输出: {result} (预期: False)")
    assert result == False
    
    # 测试用例3: answer 为 "null"
    invalid_output = {
        "answer": "null",
        "probabilities": [
            {"char": "爱", "p": 0.80}
        ]
    }
    result = validate_llm_output(invalid_output)
    print(f"✅ answer 为 'null': {result} (预期: False)")
    assert result == False
    
    # 测试用例4: 缺少 answer 字段
    invalid_output = {
        "probabilities": [
            {"char": "爱", "p": 0.80}
        ]
    }
    result = validate_llm_output(invalid_output)
    print(f"✅ 缺少 answer: {result} (预期: False)")
    assert result == False
    
    # 测试用例5: probabilities 为空列表
    invalid_output = {
        "answer": "爱",
        "probabilities": []
    }
    result = validate_llm_output(invalid_output)
    print(f"✅ probabilities 为空: {result} (预期: False)")
    assert result == False
    
    # 测试用例6: probabilities 第一项的 char 为空
    invalid_output = {
        "answer": "爱",
        "probabilities": [
            {"char": "", "p": 0.80}
        ]
    }
    result = validate_llm_output(invalid_output)
    print(f"✅ char 为空字符串: {result} (预期: False)")
    assert result == False
    
    print("\n✅ validate_llm_output() 所有测试通过！\n")


def test_is_stroke_file_valid():
    """测试 is_stroke_file_valid 函数"""
    print("\n" + "="*60)
    print("测试 is_stroke_file_valid() 函数")
    print("="*60)
    
    # 找一个实际存在的完整文件
    dataset_path = Path(__file__).parent.parent / "dataset"
    
    # 查找一个有 recognition 文件夹的字符
    test_file = None
    for char_folder in dataset_path.iterdir():
        if char_folder.is_dir() and (char_folder / "recognition").exists():
            json_files = list((char_folder / "recognition").glob("*.json"))
            if json_files:
                test_file = json_files[0]
                break
    
    if test_file:
        print(f"测试文件: {test_file}")
        result = is_stroke_file_valid(test_file)
        print(f"文件验证结果: {result}")
        
        # 读取并显示文件内容摘要
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"  字符: {data.get('character')}")
            print(f"  笔画: {data.get('stroke_name')}")
            print(f"  runs 数量: {len(data.get('runs', []))}")
            
            # 检查每个 run
            for i, run in enumerate(data.get('runs', []), 1):
                llm_output = run.get('llm_output')
                is_valid = validate_llm_output(llm_output)
                status = "✅" if is_valid else "❌"
                if is_valid and llm_output:
                    print(f"  {status} Run {i}: {llm_output.get('answer', 'N/A')}")
                else:
                    print(f"  {status} Run {i}: 无效或为 None")
        except Exception as e:
            print(f"  读取文件出错: {str(e)}")
    else:
        print("⚠️  未找到测试文件")
    
    # 测试不存在的文件
    nonexistent_file = Path("/tmp/nonexistent_stroke.json")
    result = is_stroke_file_valid(nonexistent_file)
    print(f"\n✅ 不存在的文件: {result} (预期: False)")
    assert result == False
    
    print("\n✅ is_stroke_file_valid() 测试完成！\n")


if __name__ == "__main__":
    try:
        test_validate_llm_output()
        test_is_stroke_file_valid()
        print("\n" + "="*60)
        print("🎉 所有测试通过！")
        print("="*60 + "\n")
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试错误: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
