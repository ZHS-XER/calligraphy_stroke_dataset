import fitz
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple, Dict
import re

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_INPUTS_DIR = DATA_DIR / "inputs"
INTERMEDIATE_DIR = DATA_DIR / "intermediate" / "grid_rows"

class CharacterStrokeExtractor:

    def __init__(
        self,
        pdf_path: str | Path,
        output_dir: str | Path = INTERMEDIATE_DIR,
        dpi: int = 300,
    ):

        script_dir = Path(__file__).parent
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.is_absolute():
            candidates = [
                ROOT_DIR / pdf_path_obj,
                RAW_INPUTS_DIR / pdf_path_obj.name,
                script_dir / pdf_path_obj,
            ]
            for candidate in candidates:
                if candidate.exists():
                    self.pdf_path = candidate.resolve()
                    break
            else:
                self.pdf_path = (ROOT_DIR / pdf_path_obj).resolve()
        else:
            self.pdf_path = pdf_path_obj

        output_path_obj = Path(output_dir)
        if not output_path_obj.is_absolute():
            self.output_dir = (ROOT_DIR / output_path_obj).resolve()
        else:
            self.output_dir = output_path_obj

        self.dpi = dpi
        self.zoom = dpi / 72

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_page_image(self, page) -> np.ndarray:

        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat)

        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )

        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        return img

    def detect_grid_lines(self, img: np.ndarray) -> Tuple[List[int], List[int]]:

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

        h_lines = []
        h_sum = np.sum(horizontal_lines, axis=1)
        for i, val in enumerate(h_sum):
            if val > gray.shape[1] * 100:
                h_lines.append(i)

        v_lines = []
        v_sum = np.sum(vertical_lines, axis=0)
        for i, val in enumerate(v_sum):
            if val > gray.shape[0] * 100:
                v_lines.append(i)

        h_lines = self._merge_close_lines(h_lines, threshold=10)
        v_lines = self._merge_close_lines(v_lines, threshold=10)

        return h_lines, v_lines

    def _merge_close_lines(self, lines: List[int], threshold: int = 10) -> List[int]:

        if not lines:
            return []

        lines = sorted(lines)
        merged = [lines[0]]

        for line in lines[1:]:
            if line - merged[-1] > threshold:
                merged.append(line)
            else:

                merged[-1] = (merged[-1] + line) // 2

        return merged

    def extract_character_from_grid(self, img: np.ndarray,
                                   top: int, bottom: int,
                                   left: int, right: int) -> Image.Image:

        margin = 5
        crop_img = img[top+margin:bottom-margin, left+margin:right-margin]

        return Image.fromarray(crop_img)

    def detect_character_in_cell(self, img: np.ndarray) -> bool:

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        content_ratio = np.sum(binary > 0) / binary.size

        return content_ratio > 0.01

    def extract_cells_from_row(self, row_img: np.ndarray,
                               v_lines: List[int]) -> List[Image.Image]:

        cells = []

        for i in range(len(v_lines) - 1):
            left = v_lines[i]
            right = v_lines[i + 1]

            cell_img = row_img[:, left:right]

            if self.detect_character_in_cell(cell_img):
                cells.append(Image.fromarray(cell_img))

        return cells

    def process_pdf(self, ocr_enabled: bool = False) -> Dict[str, List[str]]:

        doc = fitz.open(self.pdf_path)
        row_counter = 0
        results = {}

        print(f"Starting PDF processing: {self.pdf_path}")
        print(f"Total pages: {len(doc)}")
        print("Extraction mode: per-row cell extraction (12 cells per row)\n")

        for page_num in range(len(doc)):
            print(f"Processing page {page_num + 1}/{len(doc)}...")

            page = doc[page_num]
            img = self.extract_page_image(page)

            h_lines, v_lines = self.detect_grid_lines(img)

            print(f"  Detected {len(h_lines)} horizontal and {len(v_lines)} vertical lines")

            if len(h_lines) < 2 or len(v_lines) < 2:
                print("  Warning: insufficient grid lines detected; skipping this page")
                continue

            for row_idx in range(len(h_lines) - 1):
                top = h_lines[row_idx]
                bottom = h_lines[row_idx + 1]

                if bottom - top < 50:
                    continue

                row_img = img[top:bottom, :]

                cells = self.extract_cells_from_row(row_img, v_lines)

                if not cells:
                    continue

                row_counter += 1
                row_name = f"row_{row_counter:04d}"
                row_dir = self.output_dir / row_name
                row_dir.mkdir(exist_ok=True)

                saved_paths = []
                for cell_idx, cell_img in enumerate(cells):
                    filename = f"cell_{cell_idx:02d}.png"
                    filepath = row_dir / filename
                    cell_img.save(filepath)
                    saved_paths.append(str(filepath))

                results[row_name] = saved_paths
                print(f"  Extracted {row_name}: {len(cells)} cells")

        doc.close()

        print(f"\n✓ Done! Extracted {row_counter} rows total")
        print(f"✓ Dataset saved to: {self.output_dir}")

        return results

    def generate_summary(self, results: Dict[str, List[str]]) -> None:

        summary_path = self.output_dir / "summary.txt"

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Chinese Character Stroke Dataset Summary (row extraction)\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"PDF source: {self.pdf_path}\n")
            f.write(f"Total rows: {len(results)}\n")
            f.write("Cells per row: 12 (standard)\n\n")

            for row_name, paths in sorted(results.items()):
                f.write(f"{row_name}: {len(paths)} cells\n")

        print(f"✓ Summary file generated: {summary_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract a Chinese character stroke dataset from a calligraphy practice PDF"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Input PDF path (relative to project root, e.g., inputs/xxx.pdf)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=str(INTERMEDIATE_DIR),
        help="Output dataset directory (default: data/intermediate/grid_rows, relative to repo root)"
    )
    parser.add_argument(
        "-d", "--dpi",
        type=int,
        default=300,
        help="PDF rendering DPI (default: 300)"
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR for character detection (requires extra setup)"
    )

    args = parser.parse_args()

    extractor = CharacterStrokeExtractor(
        pdf_path=args.pdf_path,
        output_dir=args.output,
        dpi=args.dpi
    )

    results = extractor.process_pdf(ocr_enabled=args.ocr)
    extractor.generate_summary(results)

if __name__ == "__main__":
    main()
