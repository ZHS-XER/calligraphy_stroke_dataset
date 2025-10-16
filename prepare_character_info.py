#!/usr/bin/env python3
"""
预处理脚本：从 hanzi.txt 和 bihua.txt 生成字符笔画映射
为后续的智能分割提供依据
"""

import json
from pathlib import Path


def load_bihua_dict(bihua_file: str = "bihua.txt") -> dict:
    """
    从 bihua.txt 文件加载笔画字典
    
    格式: 汉字:笔画1,笔画2,笔画3,...
    返回: {字符: [笔画列表]}
    """
    bihua_dict = {}
    
    with open(bihua_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            char, strokes_str = line.split(':', 1)
            strokes = [s.strip() for s in strokes_str.split(',') if s.strip()]
            
            # 如果字符已存在，保留第一次出现的（通常是最常用的写法）
            if char not in bihua_dict:
                bihua_dict[char] = strokes
    
    return bihua_dict


def read_hanzi_list(hanzi_file: str) -> list:
    """
    从汉字文件中读取所有汉字
    
    Args:
        hanzi_file: 汉字文件路径
        
    Returns:
        汉字列表（按顺序）
    """
    with open(hanzi_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取所有汉字字符（Unicode范围：\u4e00-\u9fff）
    hanzi_list = []
    for char in content:
        if '\u4e00' <= char <= '\u9fff':
            hanzi_list.append(char)
    
    return hanzi_list


def generate_character_strokes_mapping(
    hanzi_file: str = "inputs/hanzi.txt",
    bihua_file: str = "bihua.txt",
    output_file: str = "character_strokes.json"
):
    """
    生成字符笔画映射文件
    
    Args:
        hanzi_file: 汉字列表文件
        bihua_file: 笔画数据库文件
        output_file: 输出JSON文件
    """
    print("=" * 60)
    print("📖 汉字笔画信息预处理")
    print("=" * 60)
    
    # 加载笔画字典
    print(f"\n1️⃣  加载笔画数据库: {bihua_file}")
    bihua_dict = load_bihua_dict(bihua_file)
    print(f"   ✅ 成功加载 {len(bihua_dict)} 个汉字的笔画信息")
    
    # 读取汉字列表
    print(f"\n2️⃣  读取汉字列表: {hanzi_file}")
    hanzi_list = read_hanzi_list(hanzi_file)
    print(f"   ✅ 共读取 {len(hanzi_list)} 个汉字")
    print(f"   前10个: {' '.join(hanzi_list[:10])}")
    
    # 生成映射
    print(f"\n3️⃣  生成字符笔画映射...")
    character_mapping = []
    missing_chars = []
    
    for idx, char in enumerate(hanzi_list, 1):
        if char in bihua_dict:
            strokes = bihua_dict[char]
            stroke_count = len(strokes)
            cells_needed = stroke_count + 1  # 完整字 + 各个笔画
            
            # 计算占用的行数（每行12格）
            rows_needed = (cells_needed + 11) // 12  # 向上取整
            
            char_info = {
                "index": idx,
                "character": char,
                "stroke_count": stroke_count,
                "strokes": strokes,
                "cells_needed": cells_needed,
                "rows_needed": rows_needed
            }
            character_mapping.append(char_info)
            
            # 显示跨行的字符
            if rows_needed > 1:
                print(f"   📍 {char}: {stroke_count}画 → 需要{cells_needed}格 → 跨{rows_needed}行")
        else:
            print(f"   ⚠️  {char}: 未找到笔画信息")
            missing_chars.append(char)
    
    # 保存到JSON
    print(f"\n4️⃣  保存映射文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(character_mapping, f, ensure_ascii=False, indent=2)
    
    # 统计信息
    print(f"\n" + "=" * 60)
    print(f"✅ 成功生成字符笔画映射")
    print(f"=" * 60)
    print(f"总字符数: {len(hanzi_list)}")
    print(f"成功映射: {len(character_mapping)}")
    print(f"缺失信息: {len(missing_chars)}")
    if missing_chars:
        print(f"缺失字符: {', '.join(missing_chars)}")
    
    # 跨行统计
    multi_row_chars = [c for c in character_mapping if c['rows_needed'] > 1]
    print(f"\n跨行字符: {len(multi_row_chars)} 个")
    for char_info in multi_row_chars[:10]:  # 显示前10个
        print(f"  {char_info['character']}: {char_info['stroke_count']}画 → {char_info['rows_needed']}行")
    if len(multi_row_chars) > 10:
        print(f"  ... 还有 {len(multi_row_chars) - 10} 个")
    
    print(f"\n💾 映射文件已保存: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    generate_character_strokes_mapping()
