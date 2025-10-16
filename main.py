#!/usr/bin/env python3
"""
汉字书写笔画数据集生成器
从PDF中提取每个汉字的笔画演示图片，按字符和笔画顺序组织保存
"""

import fitz  # PyMuPDF
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple, Dict
import re


class CharacterStrokeExtractor:
    """从书法练字PDF中提取汉字笔画图片"""
    
    def __init__(self, pdf_path: str, output_dir: str = "dataset", dpi: int = 300):
        """
        初始化提取器
        
        Args:
            pdf_path: PDF文件路径
            output_dir: 输出数据集目录
            dpi: PDF渲染分辨率
        """
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.zoom = dpi / 72  # PDF默认72 DPI
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_page_image(self, page) -> np.ndarray:
        """
        将PDF页面转换为图像
        
        Args:
            page: PyMuPDF页面对象
            
        Returns:
            numpy数组格式的图像
        """
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # 转换为numpy数组
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        
        # 如果是RGBA，转换为RGB
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        return img
    
    def detect_grid_lines(self, img: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        检测页面中的网格线，用于定位格子
        
        Args:
            img: 输入图像
            
        Returns:
            (水平线位置列表, 垂直线位置列表)
        """
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 二值化
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # 检测水平线
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # 检测垂直线
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # 找到水平线的y坐标
        h_lines = []
        h_sum = np.sum(horizontal_lines, axis=1)
        for i, val in enumerate(h_sum):
            if val > gray.shape[1] * 100:  # 阈值：线足够长
                h_lines.append(i)
        
        # 找到垂直线的x坐标
        v_lines = []
        v_sum = np.sum(vertical_lines, axis=0)
        for i, val in enumerate(v_sum):
            if val > gray.shape[0] * 100:  # 阈值：线足够长
                v_lines.append(i)
        
        # 合并相近的线（去除重复检测）
        h_lines = self._merge_close_lines(h_lines, threshold=10)
        v_lines = self._merge_close_lines(v_lines, threshold=10)
        
        return h_lines, v_lines
    
    def _merge_close_lines(self, lines: List[int], threshold: int = 10) -> List[int]:
        """合并距离很近的线"""
        if not lines:
            return []
        
        lines = sorted(lines)
        merged = [lines[0]]
        
        for line in lines[1:]:
            if line - merged[-1] > threshold:
                merged.append(line)
            else:
                # 取平均值
                merged[-1] = (merged[-1] + line) // 2
        
        return merged
    
    def extract_character_from_grid(self, img: np.ndarray, 
                                   top: int, bottom: int, 
                                   left: int, right: int) -> Image.Image:
        """
        从网格中提取单个格子的图像
        
        Args:
            img: 源图像
            top, bottom, left, right: 格子边界
            
        Returns:
            PIL Image对象
        """
        # 添加边距以避免裁剪到边框
        margin = 5
        crop_img = img[top+margin:bottom-margin, left+margin:right-margin]
        
        # 转换为PIL Image
        return Image.fromarray(crop_img)
    
    def detect_character_in_cell(self, img: np.ndarray) -> bool:
        """
        检测格子中是否有内容（笔画）
        
        Args:
            img: 格子图像
            
        Returns:
            是否包含内容
        """
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 计算非白色像素的比例
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        content_ratio = np.sum(binary > 0) / binary.size
        
        # 如果有足够的内容，认为是有效的
        return content_ratio > 0.01  # 至少1%的内容
    
    def extract_cells_from_row(self, row_img: np.ndarray, 
                               v_lines: List[int]) -> List[Image.Image]:
        """
        从一行中提取所有格子图片（不判断内容）
        
        Args:
            row_img: 行图像
            v_lines: 垂直线位置（列分隔符）
            
        Returns:
            格子图片列表
        """
        cells = []
        
        # 按垂直线分割格子
        for i in range(len(v_lines) - 1):
            left = v_lines[i]
            right = v_lines[i + 1]
            
            cell_img = row_img[:, left:right]
            
            # 检查格子是否有内容
            if self.detect_character_in_cell(cell_img):
                cells.append(Image.fromarray(cell_img))
        
        return cells
    
    def process_pdf(self, ocr_enabled: bool = False) -> Dict[str, List[str]]:
        """
        处理PDF文件，按行提取所有格子
        
        Args:
            ocr_enabled: 保留参数但不使用
            
        Returns:
            行到图片路径的映射
        """
        doc = fitz.open(self.pdf_path)
        row_counter = 0  # 用于给行编号
        results = {}
        
        print(f"开始处理 PDF: {self.pdf_path}")
        print(f"总页数: {len(doc)}")
        print(f"提取模式: 按行提取格子（每行12格）\n")
        
        for page_num in range(len(doc)):
            print(f"处理第 {page_num + 1}/{len(doc)} 页...")
            
            page = doc[page_num]
            img = self.extract_page_image(page)
            
            # 检测网格线
            h_lines, v_lines = self.detect_grid_lines(img)
            
            print(f"  检测到 {len(h_lines)} 条水平线, {len(v_lines)} 条垂直线")
            
            if len(h_lines) < 2 or len(v_lines) < 2:
                print("  警告: 未检测到足够的网格线，跳过此页")
                continue
            
            # 按行处理
            for row_idx in range(len(h_lines) - 1):
                top = h_lines[row_idx]
                bottom = h_lines[row_idx + 1]
                
                # 跳过太小的行
                if bottom - top < 50:
                    continue
                
                row_img = img[top:bottom, :]
                
                # 提取这一行的所有格子
                cells = self.extract_cells_from_row(row_img, v_lines)
                
                if not cells:
                    continue
                
                # 创建行文件夹
                row_counter += 1
                row_name = f"row_{row_counter:04d}"
                row_dir = self.output_dir / row_name
                row_dir.mkdir(exist_ok=True)
                
                # 保存格子图片
                saved_paths = []
                for cell_idx, cell_img in enumerate(cells):
                    filename = f"cell_{cell_idx:02d}.png"
                    filepath = row_dir / filename
                    cell_img.save(filepath)
                    saved_paths.append(str(filepath))
                
                results[row_name] = saved_paths
                print(f"  提取 {row_name}: {len(cells)} 个格子")
        
        doc.close()
        
        print(f"\n✓ 完成! 共提取 {row_counter} 行")
        print(f"✓ 数据集保存在: {self.output_dir}")
        
        return results
    
    def generate_summary(self, results: Dict[str, List[str]]) -> None:
        """
        生成数据集摘要文件
        
        Args:
            results: 提取结果
        """
        summary_path = self.output_dir / "summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"汉字笔画数据集摘要（按行提取）\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"PDF来源: {self.pdf_path}\n")
            f.write(f"总行数: {len(results)}\n")
            f.write(f"每行格子数: 12 (标准)\n\n")
            
            for row_name, paths in sorted(results.items()):
                f.write(f"{row_name}: {len(paths)} 个格子\n")
        
        print(f"✓ 摘要文件已生成: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="从书法练字PDF中提取汉字笔画数据集"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="输入PDF文件路径"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="half-done",
        help="输出数据集目录 (默认: half-done)"
    )
    parser.add_argument(
        "-d", "--dpi",
        type=int,
        default=300,
        help="PDF渲染分辨率 (默认: 300)"
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="启用OCR识别汉字（需要额外配置）"
    )
    
    args = parser.parse_args()
    
    # 创建提取器并处理
    extractor = CharacterStrokeExtractor(
        pdf_path=args.pdf_path,
        output_dir=args.output,
        dpi=args.dpi
    )
    
    results = extractor.process_pdf(ocr_enabled=args.ocr)
    extractor.generate_summary(results)


if __name__ == "__main__":
    main()
