# COT Prompt 模板

```json
[
  {
    "role": "system",
    "content": "你是一个汉字识别助手"
  },
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "示例 1：汉字书法截图（尚未完成，缺少最后一笔）。"
      },
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,{prompt_data/dataset/乔/stroke_05.png}"
        }
      },
      {
        "type": "text",
        "text": "推理过程：上部“夭”结构已完成，下部笔画尚未收势；最后一笔为竖，自上而下收束全字，形成“乔”。"
      },
      {
        "type": "text",
        "text": "正确答案：乔"
      },
      { "type": "text", "text": "---" },

      {
        "type": "text",
        "text": "示例 2：汉字书法截图（尚未完成，缺少最后一笔）。"
      },
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,{prompt_data/dataset/卧/stroke_07.png}"
        }
      },
      {
        "type": "text",
        "text": "推理过程：左右结构已显；左部“臣”完整，右部缺一点作收尾；最后一笔为点，平衡结构形成“卧”。"
      },
      {
        "type": "text",
        "text": "正确答案：卧"
      },
      { "type": "text", "text": "---" },

      {
        "type": "text",
        "text": "示例 3：汉字书法截图（尚未完成，缺少最后一笔）。"
      },
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,{prompt_data/dataset/哦/stroke_09.png}"
        }
      },
      {
        "type": "text",
        "text": "推理过程：左旁“口”已闭合，右部结构基本成形，仅缺右上角一点；最后一笔为点，用于收笔，构成“哦”。"
      },
      {
        "type": "text",
        "text": "正确答案：哦"
      },
      { "type": "text", "text": "---" },

      {
        "type": "text",
        "text": "示例 4：汉字书法截图（尚未完成，缺少最后一笔）。"
      },
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,{prompt_data/dataset/哩/stroke_09.png}"
        }
      },
      {
        "type": "text",
        "text": "推理过程：左侧“口”完成，右部“里”中田框已具，底横未写；最后一笔为横，封底成“哩”。"
      },
      {
        "type": "text",
        "text": "正确答案：哩"
      },
      { "type": "text", "text": "---" },

      {
        "type": "text",
        "text": "示例 5：汉字书法截图（尚未完成，缺少最后一笔）。"
      },
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,{prompt_data/dataset/弈/stroke_08.png}"
        }
      },
      {
        "type": "text",
        "text": "推理过程：上部“亦”结构已完，下部尚未收束；最后一笔为竖，自上而下完成整体对称，形成“弈”。"
      },
      {
        "type": "text",
        "text": "正确答案：弈"
      },
      { "type": "text", "text": "---" },

      {
        "type": "text",
        "text": "示例 6：汉字书法截图（尚未完成，缺少最后一笔）。"
      },
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,{prompt_data/dataset/拓/stroke_07.png}"
        }
      },
      {
        "type": "text",
        "text": "推理过程：左旁“扌”三笔齐全，右部结构明确，底部留空；最后一笔为横，封底成“拓”。"
      },
      {
        "type": "text",
        "text": "正确答案：拓"
      },
      { "type": "text", "text": "---" },

      {
        "type": "text",
        "text": "示例 7：汉字书法截图（尚未完成，缺少最后一笔）。"
      },
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,{prompt_data/dataset/渗/stroke_10.png}"
        }
      },
      {
        "type": "text",
        "text": "推理过程：左旁三点水完整，右部笔画交叉已现；最后一笔为撇，自右上向左下收势，完成“渗”。"
      },
      {
        "type": "text",
        "text": "正确答案：渗"
      },
      { "type": "text", "text": "---" },

      {
        "type": "text",
        "text": "示例 8：汉字书法截图（尚未完成，缺少最后一笔）。"
      },
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,{prompt_data/dataset/狡/stroke_08.png}"
        }
      },
      {
        "type": "text",
        "text": "推理过程：左旁“犭”三笔齐备，右部“交”结构清晰，右下未收；最后一笔为捺，从上右向下右收尾，完成“狡”。"
      },
      {
        "type": "text",
        "text": "正确答案：狡"
      },
      { "type": "text", "text": "---" },

      {
        "type": "text",
        "text": "任务：识别单帧书法图像中的汉字（该帧信息已足够唯一确定该字）。\n\n上下文：\n- 本图像是汉字书法教学视频的截图。\n- 当前帧包含足以判断目标汉字的笔画与结构。\n- 此时还剩1个笔划没有写完。\n\n输出规范（务必严格遵守）：\n- 只输出一个 JSON 对象，且不包含任何多余文字、换行前缀或注释。\n- 字段与取值：\n  \"answer\": \"<单个汉字>\",                       // 最终判定（必须与概率列表首项一致）\n  \"probabilities\": [                            // 按概率降序，长度 3\n    {\"char\":\"<汉字1>\",\"p\":0.xx},\n    {\"char\":\"<汉字2>\",\"p\":0.xx},\n    {\"char\":\"<汉字3>\",\"p\":0.xx}\n  ],\n  \"rationale\": \"<推断依据>\"\n\n严格格式要求：\n- 概率 p 为 0–1 之间的小数，保留两位；所有 p 之和必须等于 1.00（四舍五入后）。\n- 若 \"answer\" ≠ probabilities[0].char，则令 \"answer\" = probabilities[0].char。\n- 若可能的汉字数 < 3，用 null 代替汉字，概率则用 0.00 填充。\n- 仅可输出中文汉字作为候选，不得输出词语、拼音或符号。\n\n校准指引（内部执行，最终仅输出 JSON）：\n- 先为 Top-K 候选分配相对打分 s_i ∈ {1..9}，再归一化为概率 p_i = s_i / Σ s_i。\n- 若最高概率 < 0.50，仍给出最可能答案。\n- 在 \"rationale\" 中简述推断依据以及歧义来源。\n\n现在输出符合上述规范的 JSON。"
      },
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,{dataset/桥/stroke_09.png}"
        }
      }
    ]
  }
]
```
