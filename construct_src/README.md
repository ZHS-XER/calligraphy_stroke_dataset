# 数据集构建脚本

此目录包含构建汉字笔画数据集的所有脚本。

## 脚本列表

1. **prepare_character_info.py** - 生成字符笔画映射
   - 读取 `inputs/hanzi.txt` 和 `bihua.txt`
   - 生成 `character_strokes.json`

2. **main.py** - 从 PDF 提取格子图
   - 输入: PDF 文件
   - 输出: `half-done/row_XXXX/cell_YY.png`

3. **allocate_characters.py** - 按字分配格子
   - 输入: `character_strokes.json` + `half-done/`
   - 输出: `dataset/<字>/stroke_*.png`

4. **binarize_dataset.py** - 图片二值化
   - 输入/输出: `dataset/`（原地修改）
   - 将灰色像素转为纯黑

5. **analyze_strokes.py** - LLM 笔画验证
   - 输入: `dataset/`
   - 输出: `stroke_analysis_results.json`

## 使用方法

所有脚本都从项目根目录运行：

```bash
# 从项目根目录运行
uv run construct_src/prepare_character_info.py
uv run construct_src/main.py "inputs/xxx.pdf"
uv run construct_src/allocate_characters.py
uv run construct_src/binarize_dataset.py -d dataset --execute
uv run construct_src/analyze_strokes.py -d dataset
```

## 路径说明

- 所有脚本使用相对于脚本所在目录的路径
- 输入/输出文件都位于项目根目录
- `../` 表示项目根目录
- `../inputs/` 表示输入文件目录
- `../half-done/` 表示中间结果目录
- `../dataset/` 表示最终数据集目录
