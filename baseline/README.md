# Baseline 识别脚本（简化版）

纯粹收集大模型的 JSON 输出，不做任何额外处理和统计。

## 核心功能

✅ **只做一件事**: 收集所有大模型的 JSON 输出  
✅ **断点续传**: 中断后自动继续  
✅ **失败重试**: 最多重试 100 次  
✅ **并行处理**: 多线程加速  

## 使用方法

### 快速开始

```bash
cd baseline

# 测试（处理前 2 个字符）
uv run recognize_strokes.py --test 2

# 正式运行
uv run recognize_strokes.py
```

### 参数说明

```bash
-d, --dataset      数据集目录 (默认: ../dataset)
-o, --output       输出文件 (默认: baseline_llm_outputs.json)
-w, --max-workers  并行线程数 (默认: 5)
--no-resume        不使用断点续传
--test N           测试模式，只处理前 N 个字符
```

### 示例

```bash
# 使用 10 个线程
uv run recognize_strokes.py -w 10

# 输出到自定义文件
uv run recognize_strokes.py -o my_results.json

# 重新开始
uv run recognize_strokes.py --no-resume
```

## 输出格式

输出文件是一个纯粹的 JSON 数组，包含所有字符的大模型输出：

```json
[
  {
    "character": "一",
    "strokes": [
      {
        "stroke_name": "stroke_01",
        "image_path": "dataset/一/stroke_01.png",
        "llm_output": {
          "answer": "一",
          "probabilities": [
            {"char": "一", "p": 1.0},
            {"char": null, "p": 0.0},
            {"char": null, "p": 0.0}
          ],
          "rationale": "推断依据..."
        }
      }
    ]
  },
  {
    "character": "丁",
    "strokes": [
      {
        "stroke_name": "stroke_01",
        "image_path": "dataset/丁/stroke_01.png",
        "llm_output": {
          "answer": "一",
          "probabilities": [...],
          "rationale": "..."
        }
      },
      {
        "stroke_name": "stroke_02",
        "image_path": "dataset/丁/stroke_02.png",
        "llm_output": {
          "answer": "丁",
          "probabilities": [...],
          "rationale": "..."
        }
      }
    ]
  }
]
```

## 工作流程

```
遍历 dataset/ 目录
    ↓
对每个字符文件夹
    ↓
获取所有 stroke_*.png（排除 stroke_00.png）
    ↓
逐个发送给 API
    ↓
收集大模型的 JSON 输出
    ↓
保存到数组中
    ↓
每 10 个字符保存一次
    ↓
输出到 JSON 文件
```

## 断点续传

脚本会自动：
- 读取已有的输出文件
- 跳过已处理的字符
- 从中断处继续
- 每 10 个字符保存一次进度

随时按 `Ctrl+C` 中断，下次运行会自动继续。

## 预期结果

假设数据集有 2488 个字，平均每字 5 笔：
- **总请求数**: ~12,440 次
- **输出文件**: 包含所有 12,440 个大模型 JSON 响应
- **文件大小**: 约 50-100 MB
- **耗时**: 1-2 小时（取决于网络和线程数）

## 注意事项

1. **输出文件很大**: 包含所有大模型的完整响应
2. **只收集数据**: 不做任何统计和分析
3. **保留失败记录**: 失败的笔画会记录 `llm_output: null` 和 `error` 字段
4. **相对路径**: 图片路径使用相对于项目根目录的路径

## 后续分析

收集完数据后，你可以：
- 用其他脚本分析 `llm_output` 中的 `answer`
- 统计 `probabilities` 的分布
- 分析 `rationale` 的模式
- 计算识别准确率
- 提取特定字段做进一步处理

这个脚本只负责收集原始数据！
