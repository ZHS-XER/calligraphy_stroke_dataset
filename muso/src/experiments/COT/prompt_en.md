# COT Prompt Template (English)

```json
[
  {
    "role": "system",
    "content": "You are a Chinese character recognition assistant."
  },
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "Example 1: Calligraphy screenshot (incomplete, missing the final stroke)."
      },
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,{prompt_data/dataset/乔/stroke_05.png}"
        }
      },
      {
        "type": "text",
        "text": "Reasoning: The upper “夭” structure is complete; the lower part hasn’t closed. The final stroke is a vertical that closes the whole character from top to bottom, forming “乔”."
      },
      {
        "type": "text",
        "text": "Correct answer: 乔"
      },
      { "type": "text", "text": "---" },

      {
        "type": "text",
        "text": "Example 2: Calligraphy screenshot (incomplete, missing the final stroke)."
      },
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,{prompt_data/dataset/卧/stroke_07.png}"
        }
      },
      {
        "type": "text",
        "text": "Reasoning: The left “臣” is complete and the overall left-right structure is clear; the right side needs one dot to finish. The final stroke is a dot to balance the structure, forming “卧”."
      },
      {
        "type": "text",
        "text": "Correct answer: 卧"
      },
      { "type": "text", "text": "---" },

      {
        "type": "text",
        "text": "Example 3: Calligraphy screenshot (incomplete, missing the final stroke)."
      },
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,{prompt_data/dataset/哦/stroke_09.png}"
        }
      },
      {
        "type": "text",
        "text": "Reasoning: The left “口” is closed; the right structure is mostly formed but missing the top-right dot. The final stroke is that dot to finish the character “哦”."
      },
      {
        "type": "text",
        "text": "Correct answer: 哦"
      },
      { "type": "text", "text": "---" },

      {
        "type": "text",
        "text": "Example 4: Calligraphy screenshot (incomplete, missing the final stroke)."
      },
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,{prompt_data/dataset/哩/stroke_09.png}"
        }
      },
      {
        "type": "text",
        "text": "Reasoning: The left “口” is done; in the right “里”, the field frame is present but the bottom horizontal is missing. The final stroke is a horizontal to close the bottom, forming “哩”."
      },
      {
        "type": "text",
        "text": "Correct answer: 哩"
      },
      { "type": "text", "text": "---" },

      {
        "type": "text",
        "text": "Example 5: Calligraphy screenshot (incomplete, missing the final stroke)."
      },
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,{prompt_data/dataset/弈/stroke_08.png}"
        }
      },
      {
        "type": "text",
        "text": "Reasoning: The upper “亦” is complete; the lower part isn’t closed yet. The final stroke is a vertical from top to bottom to finish the symmetry, forming “弈”."
      },
      {
        "type": "text",
        "text": "Correct answer: 弈"
      },
      { "type": "text", "text": "---" },

      {
        "type": "text",
        "text": "Example 6: Calligraphy screenshot (incomplete, missing the final stroke)."
      },
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,{prompt_data/dataset/拓/stroke_07.png}"
        }
      },
      {
        "type": "text",
        "text": "Reasoning: The left “扌” three strokes are complete; the right structure is clear but the bottom is open. The final stroke is a horizontal to close the bottom, forming “拓”."
      },
      {
        "type": "text",
        "text": "Correct answer: 拓"
      },
      { "type": "text", "text": "---" },

      {
        "type": "text",
        "text": "Example 7: Calligraphy screenshot (incomplete, missing the final stroke)."
      },
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,{prompt_data/dataset/渗/stroke_10.png}"
        }
      },
      {
        "type": "text",
        "text": "Reasoning: The left three-dots water is complete; the right strokes already cross. The final stroke is a left-falling stroke from upper right to lower left to finish “渗”."
      },
      {
        "type": "text",
        "text": "Correct answer: 渗"
      },
      { "type": "text", "text": "---" },

      {
        "type": "text",
        "text": "Example 8: Calligraphy screenshot (incomplete, missing the final stroke)."
      },
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,{prompt_data/dataset/狡/stroke_08.png}"
        }
      },
      {
        "type": "text",
        "text": "Reasoning: The left “犭” is complete; the right “交” is clear but the lower-right isn’t finished. The final stroke is a pressing stroke (捺) from upper right to lower right to complete “狡”."
      },
      {
        "type": "text",
        "text": "Correct answer: 狡"
      },
      { "type": "text", "text": "---" },

      {
        "type": "text",
        "text": "Task: Identify the Chinese character in a single calligraphy frame (the frame already uniquely determines the character).\n\nContext:\n- The image is a screenshot from a Chinese calligraphy teaching video.\n- The current frame contains enough strokes/structure to decide the target character.\n- There is 1 stroke left unfinished.\n\nOutput spec (must follow):\n- Output a single JSON object with no extra text, prefixes, or Markdown.\n- Fields:\n  \"answer\": \"<single Chinese character>\",                       // final decision (must match the first probability entry)\n  \"probabilities\": [                            // sorted by probability desc, length 3\n    {\"char\":\"<char1>\",\"p\":0.xx},\n    {\"char\":\"<char2>\",\"p\":0.xx},\n    {\"char\":\"<char3>\",\"p\":0.xx}\n  ],\n  \"rationale\": \"<brief reasoning>\"\n\nStrict rules:\n- Probability p is 0–1 with two decimals; all p sum to 1.00 (after rounding).\n- If \"answer\" ≠ probabilities[0].char, set \"answer\" = probabilities[0].char.\n- If fewer than 3 candidates, use null for the character and 0.00 for the probability.\n- Only output Chinese characters as candidates; no words, pinyin, or symbols.\n\nCalibration (internal; still output only JSON):\n- First assign relative scores s_i ∈ {1..9} to Top-K candidates, then normalize to probabilities p_i = s_i / Σ s_i.\n- If the top probability < 0.50, still output the most likely answer.\n- In \"rationale\", briefly explain evidence and sources of ambiguity.\n\nNow output the JSON that follows these rules."
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
