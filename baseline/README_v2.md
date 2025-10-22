# 笔画识别系统 v2.0

## 🎯 核心改进

### 版本对比

| 特性 | v1.0 | v2.0 (当前) |
|------|------|-------------|
| 每张图运行次数 | 1次 | **5次** |
| 输出位置 | 单个大 JSON | **每个字符的 `recognition/` 文件夹** |
| 文件组织 | 集中式 | **分布式（每张图一个文件）** |
| 断点续传 | 基于字符 | **基于文件（更精细）** |

## 📁 输出结构

```
dataset/
├── 一/
│   ├── stroke_00.png
│   ├── stroke_01.png
│   └── recognition/              ← 新增
│       └── stroke_01.json        ← 每张图一个文件
├── 丁/
│   ├── stroke_00.png
│   ├── stroke_01.png
│   ├── stroke_02.png
│   └── recognition/
│       ├── stroke_01.json
│       └── stroke_02.json
...
```

## 📊 JSON 格式

每个 `stroke_XX.json` 文件包含：

```json
{
  "character": "一",
  "stroke_name": "stroke_01",
  "image_path": "dataset/一/stroke_01.png",
  "runs": [
    {
      "run_id": 1,
      "llm_output": {
        "answer": "一",
        "probabilities": [...],
        "rationale": "..."
      }
    },
    {
      "run_id": 2,
      "llm_output": {...}
    },
    // ... 共5次
  ]
}
```

## 🚀 使用方法

### 测试模式
```bash
# 测试前 10 个字符
uv run recognize_strokes.py --test 10
```

### 完整运行
```bash
# 处理所有 2488 个字符
uv run recognize_strokes.py

# 自定义并行数（默认 5）
uv run recognize_strokes.py --max-workers 10

# 不使用断点续传（从头开始）
uv run recognize_strokes.py --no-resume
```

## ⚙️ 核心机制

### 1. 多次运行 (5次)
- 每张笔画图片运行 **5 次独立识别**
- 捕捉 LLM 输出的随机性
- 用于后续一致性分析

### 2. 分布式存储
- 每张图的识别结果存为**独立 JSON 文件**
- 文件路径：`dataset/{字符}/recognition/{笔画名}.json`
- 易于增量更新和并行处理

### 3. 智能断点续传
- 检查每个字符的 `recognition/` 文件夹
- 自动跳过已完成的笔画
- 支持任意时刻中断和恢复

### 4. 重试机制
- 每次 API 调用最多重试 100 次
- 指数退避：0.5s → 1s → 2s → 4s...（最大 10s）
- 网络错误自动恢复

## 📈 数据规模

- **字符数**: 2,488
- **平均笔画数**: ~5
- **总 API 调用**: ~12,440 × 5 = **~62,200 次**
- **预计耗时**: 5-10 小时（5 并发）

## 💡 优势

### 数据分析友好
- ✅ 每个字符独立，易于筛选
- ✅ 每张图独立，易于抽样
- ✅ 5次运行，可分析一致性

### 容错性强
- ✅ 单个文件失败不影响其他
- ✅ 可随时中断，精确恢复
- ✅ 增量处理，逐步完善

### 后续分析灵活
```python
# 示例：读取某个字符的所有识别结果
import json
from pathlib import Path

char_dir = Path("dataset/一/recognition")
for stroke_file in char_dir.glob("*.json"):
    with open(stroke_file) as f:
        data = json.load(f)
        # 分析 5 次运行的结果
        for run in data['runs']:
            print(run['llm_output']['answer'])
```

## 🔍 进度监控

脚本会生成 `baseline_summary.json` 摘要文件：
```json
{
  "total_characters": 2488,
  "processed_characters": 100,
  "processing_time_seconds": 3600,
  "results": [
    {
      "character": "一",
      "strokes_processed": 1
    },
    ...
  ]
}
```

## 📝 注意事项

1. **存储空间**: 每张图 5 次运行，JSON 文件较大（~3-4 KB/文件）
2. **API 配额**: 62,200 次调用，确保 API 余额充足
3. **网络稳定**: 长时间运行，建议稳定网络环境
4. **并发控制**: 默认 5 并发，可根据 API 限制调整

## 🎉 改进效果

相比 v1.0：
- ✅ **鲁棒性提升**: 5 次运行捕捉 LLM 不确定性
- ✅ **组织性增强**: 分布式存储，逻辑清晰
- ✅ **可分析性**: 支持一致性、置信度等分析
- ✅ **容错性好**: 精细化断点续传
