> The reorganized MUSO source, dataset, and setup instructions are in [muso/](muso/).

# 汉字笔画数据集

- 由https://www.an2.net/zi/获取自动生成的带笔顺字帖 pdf 文件
- 由https://github.com/DongSky/zhHanSequence/blob/master/bihua.txt获取汉字笔顺查找表
（可能不齐全 必要时手动补齐）
# 汉字笔画数据集（最简用法）

从练字 PDF 提取每行格子图，再按笔画信息分配到每个字的文件夹。

## 项目结构

```
.
├── inputs/                  # 输入文件
│   ├── hanzi.txt           # 汉字清单（按顺序）
│   └── *.pdf               # 练字 PDF
├── bihua.txt               # 笔画字典（20,795+ 字）
├── character_strokes.json  # 生成的笔画映射
├── construct_src/          # 数据集构建脚本
│   ├── prepare_character_info.py
│   ├── main.py
│   ├── allocate_characters.py
│   ├── binarize_dataset.py
│   └── analyze_strokes.py
├── half-done/              # 中间结果（每行12格）
│   ├── row_0000/
│   │   ├── cell_00.png
│   │   └── ...
│   └── summary.txt
├── dataset/                # 最终数据集 ⭐
│   ├── 一/
│   │   ├── stroke_00.png  # 整字图
│   │   ├── stroke_01.png  # 第1笔
│   │   ├── stroke_info.json
│   │   └── ...
│   ├── 二/
│   └── ...
└── README.md
```

## 最终数据集效果

每个字符文件夹（如 `dataset/一/`）包含：
- **stroke_00.png**：完整汉字图片
- **stroke_01.png ~ stroke_XX.png**：按笔顺排列的各笔画图片
- **stroke_info.json**：笔画元数据
  ```json
  {
    "character": "一",
    "stroke_count": 1,
    "strokes": ["横"],
    "total_images": 2
  }
  ```

所有图片经二值化处理后，笔画为纯黑色（0），背景为纯白色（255），适用于 OCR、笔画识别等机器学习任务。


1) 生成笔画映射（基于 hanzi.txt + bihua.txt）

```zsh
uv run construct_src/prepare_character_info.py
```

- 使用文件：`inputs/hanzi.txt`、`bihua.txt`
- 产出文件：`character_strokes.json`（包含每字的笔画数、笔画名、需要格子数与行数）
- 它做了什么：读取你的汉字清单，查 `bihua.txt` 的标准笔画；计算 `cells_needed = 笔画数+1`、`rows_needed = ceil(cells/12)`，为后续分配做定位。

2) 从 PDF 提取"每行12格"的中间结果

```zsh
uv run construct_src/main.py "inputs/你的文件.pdf"
```

- 使用文件：输入 PDF（位于 `inputs/`）
- 产出目录：`half-done/row_XXXX/cell_YY.png`（每行一个文件夹，每格一张图）
- 附加：`half-done/summary.txt`（每行格子计数）
- 它做了什么：把 PDF 渲染成图，用 OpenCV 检测水平/垂直网格线；按网格裁剪一行12张格子图（空白格会被过滤）。

3) 按字分配格子，生成最终数据集

```zsh
uv run construct_src/allocate_characters.py
```

- 使用文件/目录：`character_strokes.json`、`half-done/`
- 产出目录：`dataset/<字>/`
	- `stroke_00.png`：整字图
	- `stroke_01.png ...`：各笔画图（顺序与 `stroke_info.json` 一致）
	- `stroke_info.json`：笔画名称与数量
- 它做了什么：按 `character_strokes.json` 顺序，每个字从"新行的第1格"开始拿图；若需要跨行则自动衔接下一行，保证一行只属于一个字。

4)（可选）用大模型复核笔画

```zsh
uv run construct_src/analyze_strokes.py -d dataset
```

- 使用目录：`dataset/`
- 产出文件：`stroke_analysis_results.json`
- 它做了什么：对 `dataset/<字>/` 逐字并发请求 LLM，得到笔画名与数量的 JSON 结果；支持断点续传。

5)（可选）图片二值化处理

```zsh
uv run construct_src/binarize_dataset.py -d dataset --execute
```

- 使用目录：`dataset/`（原地修改全部 `stroke_*.png`）
- 参数说明：
	- `--execute`：实际执行修改（不加则为预览模式）
	- `--test N`：仅处理前 N 个字（用于测试）
	- `--threshold 240`：二值化阈值（默认 240，像素值 >240→白色 255，≤240→黑色 0）
- 它做了什么：将图片中的灰色像素点全部转为纯黑色（0），白色背景保留（255），确保笔画颜色统一，适用于模型训练或进一步分析。

2) 从 PDF 提取“每行12格”的中间结果

```zsh
uv run main.py "inputs/你的文件.pdf"
```

- 使用文件：输入 PDF（位于 `inputs/`）
- 产出目录：`half-done/row_XXXX/cell_YY.png`（每行一个文件夹，每格一张图）
- 附加：`half-done/summary.txt`（每行格子计数）
- 它做了什么：把 PDF 渲染成图，用 OpenCV 检测水平/垂直网格线；按网格裁剪一行12张格子图（空白格会被过滤）。

3) 按字分配格子，生成最终数据集

```zsh
uv run allocate_characters.py
```

- 使用文件/目录：`character_strokes.json`、`half-done/`
- 产出目录：`dataset/<字>/`
	- `stroke_00.png`：整字图
	- `stroke_01.png ...`：各笔画图（顺序与 `stroke_info.json` 一致）
	- `stroke_info.json`：笔画名称与数量
- 它做了什么：按 `character_strokes.json` 顺序，每个字从“新行的第1格”开始拿图；若需要跨行则自动衔接下一行，保证一行只属于一个字。

4)（可选）用大模型复核笔画

```zsh
uv run analyze_strokes.py -d dataset
```

- 使用目录：`dataset/`
- 产出文件：`stroke_analysis_results.json`
- 它做了什么：对 `dataset/<字>/` 逐字并发请求 LLM，得到笔画名与数量的 JSON 结果；支持断点续传。

5)（可选）图片二值化处理

```zsh
uv run binarize_dataset.py -d dataset --execute
```

- 使用目录：`dataset/`（原地修改全部 `stroke_*.png`）
- 参数说明：
	- `--execute`：实际执行修改（不加则为预览模式）
	- `--test N`：仅处理前 N 个字（用于测试）
	- `--threshold 240`：二值化阈值（默认 240，像素值 >240→白色 255，≤240→黑色 0）
- 它做了什么：将图片中的灰色像素点全部转为纯黑色（0），白色背景保留（255），确保笔画颜色统一，适用于模型训练或进一步分析。

## 小贴士

- 每行固定 12 格；每字需要格子 = 笔画数 + 1；跨行自动处理（每字从新行开始）。
- 重新开始：

```zsh
rm -rf half-done dataset character_strokes.json stroke_analysis_results*.json
```

