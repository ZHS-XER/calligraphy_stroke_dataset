#!/usr/bin/env python3
"""
图片二值化工具：将 dataset 中所有图片的灰色像素转为纯黑色
保持白色背景不变，使笔画变为纯黑
"""

import argparse
from pathlib import Path
from PIL import Image
import numpy as np


def binarize_image(img: Image.Image, threshold: int = 240) -> Image.Image:
    """
    将图片二值化：灰色像素变黑，白色背景保留
    
    Args:
        img: 输入图片
        threshold: 阈值，大于此值认为是白色背景（默认240）
        
    Returns:
        二值化后的图片
    """
    # 转为灰度图
    if img.mode != 'L':
        img = img.convert('L')
    
    # 转为 numpy 数组
    img_array = np.array(img)
    
    # 二值化：> threshold 的保留为白色(255)，其余变为黑色(0)
    binary_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
    
    # 转回 PIL Image
    return Image.fromarray(binary_array, mode='L')


def process_dataset(dataset_dir: str = "dataset", 
                   threshold: int = 240,
                   dry_run: bool = False) -> None:
    """
    处理数据集中的所有图片
    
    Args:
        dataset_dir: 数据集目录
        threshold: 二值化阈值
        dry_run: 如果为 True，只统计不修改
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ 错误: 数据集目录不存在 - {dataset_dir}")
        return
    
    # 查找所有字符文件夹
    char_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    # 统计
    total_images = 0
    processed_images = 0
    
    print(f"{'='*60}")
    print(f"🎨 图片二值化处理")
    print(f"{'='*60}")
    print(f"数据集目录: {dataset_dir}")
    print(f"二值化阈值: {threshold}")
    print(f"字符数量: {len(char_folders)}")
    
    if dry_run:
        print(f"⚠️  预览模式（不会修改文件）")
    
    print(f"{'='*60}\n")
    
    # 遍历所有字符文件夹
    for idx, char_folder in enumerate(char_folders, 1):
        # 查找所有 PNG 图片
        image_files = sorted(char_folder.glob("stroke_*.png"))
        
        if idx % 100 == 0 or idx == 1:
            print(f"处理进度: {idx}/{len(char_folders)} ({char_folder.name})")
        
        for img_file in image_files:
            total_images += 1
            
            try:
                # 读取图片
                img = Image.open(img_file)
                
                # 二值化
                binary_img = binarize_image(img, threshold=threshold)
                
                # 保存（如果不是 dry_run）
                if not dry_run:
                    binary_img.save(img_file)
                
                processed_images += 1
                
            except Exception as e:
                print(f"\n⚠️  处理失败: {img_file}")
                print(f"   错误: {str(e)}")
    
    # 输出统计
    print(f"\n{'='*60}")
    print(f"✅ 处理完成")
    print(f"{'='*60}")
    print(f"总图片数: {total_images}")
    print(f"成功处理: {processed_images}")
    
    if dry_run:
        print(f"\n💡 预览完成，使用 --execute 执行实际修改")
    else:
        print(f"\n✅ 所有图片已二值化（灰色→黑色，白色保留）")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="将数据集中所有图片二值化：灰色像素变黑，白色背景保留"
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="dataset",
        help="数据集目录 (默认: dataset)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=int,
        default=240,
        help="二值化阈值，大于此值认为是白色背景 (默认: 240)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="执行修改（不加此参数只预览）"
    )
    parser.add_argument(
        "--test",
        type=int,
        metavar="N",
        help="测试模式：只处理前 N 个字符"
    )
    
    args = parser.parse_args()
    
    # 测试模式
    if args.test:
        print(f"\n⚠️  测试模式：只处理前 {args.test} 个字符\n")
        dataset_path = Path(args.dataset)
        char_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir()])[:args.test]
        
        processed = 0
        for char_folder in char_folders:
            image_files = sorted(char_folder.glob("stroke_*.png"))
            print(f"处理 {char_folder.name}: {len(image_files)} 张图片")
            
            for img_file in image_files:
                try:
                    img = Image.open(img_file)
                    binary_img = binarize_image(img, threshold=args.threshold)
                    
                    if args.execute:
                        binary_img.save(img_file)
                    
                    processed += 1
                except Exception as e:
                    print(f"  ❌ 失败: {img_file.name} - {str(e)}")
        
        print(f"\n✅ 测试完成，处理了 {processed} 张图片")
        if not args.execute:
            print(f"💡 使用 --execute 执行实际修改")
    else:
        # 完整处理
        process_dataset(
            dataset_dir=args.dataset,
            threshold=args.threshold,
            dry_run=not args.execute
        )


if __name__ == "__main__":
    main()
