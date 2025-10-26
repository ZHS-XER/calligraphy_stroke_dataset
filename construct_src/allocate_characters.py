#!/usr/bin/env python3
"""
智能分配脚本：根据 character_strokes.json 将 row 中的格子分配给字符
支持跨行字符的处理
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict


def load_character_mapping(mapping_file: str = "../character_strokes.json") -> List[Dict]:
    """
    加载字符笔画映射
    
    Returns:
        字符信息列表
    """
    # 解析相对于脚本所在目录的路径
    script_dir = Path(__file__).parent
    mapping_path = script_dir / mapping_file
    
    with open(mapping_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def allocate_characters_to_rows(
    row_dataset_dir: str = "../half-done",
    character_mapping_file: str = "../character_strokes.json",
    output_dir: str = "../dataset"
):
    """
    将行格子分配给字符
    
    Args:
        row_dataset_dir: 按行提取的数据集目录（相对于脚本目录）
        character_mapping_file: 字符笔画映射文件（相对于脚本目录）
        output_dir: 输出的字符数据集目录（相对于脚本目录）
    """
    print("=" * 60)
    print("🎯 智能字符分配")
    print("=" * 60)
    
    # 解析路径
    script_dir = Path(__file__).parent
    
    # 加载字符映射
    print(f"\n1️⃣  加载字符映射: {character_mapping_file}")
    char_mapping = load_character_mapping(character_mapping_file)
    print(f"   ✅ 成功加载 {len(char_mapping)} 个字符")
    
    # 扫描行文件夹
    print(f"\n2️⃣  扫描行数据: {row_dataset_dir}")
    row_path = script_dir / row_dataset_dir
    
    if not row_path.exists():
        print(f"   ❌ 错误: 数据集目录不存在: {row_path}")
        return
    
    row_folders = sorted([d for d in row_path.iterdir() if d.is_dir() and d.name.startswith('row_')])
    print(f"   ✅ 找到 {len(row_folders)} 行")
    
    # 创建输出目录
    output_path = script_dir / output_dir
    output_path.mkdir(exist_ok=True)
    print(f"\n3️⃣  输出目录: {output_path}")
    
    # 开始分配
    print(f"\n4️⃣  开始分配字符...")
    print(f"{'='*60}")
    
    current_row_idx = 0  # 当前行索引（从0开始）
    
    success_count = 0
    skip_count = 0
    
    for char_info in char_mapping:
        char = char_info['character']
        cells_needed = char_info['cells_needed']
        stroke_count = char_info['stroke_count']
        strokes = char_info['strokes']
        rows_needed = char_info['rows_needed']
        
        # 创建字符文件夹
        char_dir = output_path / char
        char_dir.mkdir(exist_ok=True)
        
        # 收集需要的格子（每个字从新行开始，一行只属于一个字）
        collected_cells = []
        
        # 检查是否有足够的行
        if current_row_idx + rows_needed > len(row_folders):
            print(f"   ⚠️  {char}: 超出可用行数（需要{rows_needed}行，剩余{len(row_folders) - current_row_idx}行），跳过")
            skip_count += 1
            current_row_idx += rows_needed  # 即使跳过也要移动指针
            continue
        
        # 从当前行开始，按需跨行提取格子
        cell_idx_in_char = 0
        for row_offset in range(rows_needed):
            row_folder = row_folders[current_row_idx + row_offset]
            
            # 计算这一行要取多少格子
            cells_in_this_row = min(12, cells_needed - cell_idx_in_char)
            
            for cell_idx in range(cells_in_this_row):
                cell_file = row_folder / f"cell_{cell_idx:02d}.png"
                
                if cell_file.exists():
                    collected_cells.append(cell_file)
                else:
                    print(f"   ⚠️  {char}: 缺少格子 {row_folder.name}/cell_{cell_idx:02d}.png")
                
                cell_idx_in_char += 1
        
        # 如果收集到了所有格子，复制到字符文件夹
        if len(collected_cells) == cells_needed:
            # 第一个是完整字，后续是笔画
            for idx, cell_file in enumerate(collected_cells):
                target_file = char_dir / f"stroke_{idx:02d}.png"
                shutil.copy2(cell_file, target_file)
            
            # 保存笔画信息
            stroke_info = {
                "character": char,
                "stroke_count": stroke_count,
                "strokes": strokes,
                "total_images": len(collected_cells)
            }
            with open(char_dir / "stroke_info.json", 'w', encoding='utf-8') as f:
                json.dump(stroke_info, f, ensure_ascii=False, indent=2)
            
            # 显示信息
            if rows_needed > 1:
                print(f"   ✅ {char}: {stroke_count}画 → {cells_needed}格 → 跨{rows_needed}行 (row_{current_row_idx+1:04d}~{current_row_idx+rows_needed:04d})")
            else:
                print(f"   ✅ {char}: {stroke_count}画 → {cells_needed}格 → row_{current_row_idx+1:04d}")
            
            success_count += 1
        else:
            print(f"   ❌ {char}: 只收集到{len(collected_cells)}/{cells_needed}格")
            skip_count += 1
        
        # 移动到下一个字的起始行（每个字独占行）
        current_row_idx += rows_needed
    
    # 总结
    print(f"{'='*60}")
    print(f"\n✅ 分配完成")
    print(f"{'='*60}")
    print(f"总字符数: {len(char_mapping)}")
    print(f"成功分配: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"输出目录: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    allocate_characters_to_rows()
