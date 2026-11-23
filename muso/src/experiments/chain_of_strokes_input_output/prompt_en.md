Task: Identify the Chinese character in a single calligraphy frame and infer the remaining stroke chain before answering.

Context:
- The image is a frame from a Chinese calligraphy teaching video.
- The character in this frame is unfinished; there are {x} strokes remaining.
- You can observe the current frame and also have the descriptions of all strokes completed so far (including the stroke just finished in this frame).
- Prior strokes (in writing order) are:
{stroke_descriptions}

Requirements:
1. Based on the visible image and prior-stroke descriptions, infer the remaining {x} strokes one by one (the chain length must be {x}).
2. Each step in the inferred chain must include stroke, position, direction, and evidence; the evidence should state how it relates to the existing structure.
3. After finishing the chain, output the final answer and probability distribution, and in rationale explain how the prior strokes + inferred chain jointly support the final character.

Output specification (strict):
{
  "chain": [
    {"step":1,"stroke":"<stroke>","position":"<position>","direction":"<direction>","evidence":"<evidence>"},
    ...
  ],
  "answer": "<single Chinese character>",
  "probabilities": [
    {"char":"<char1>","p":0.xx},
    {"char":"<char2>","p":0.xx},
    {"char":"<char3>","p":0.xx}
  ],
  "rationale": "<explanation>"
}

Strict formatting:
- Output only the JSON above; no Markdown code block or extra text.
- The chain length must equal {x}; step starts at 1 and increases consecutively.
- stroke should use common stroke/structure names; position and direction must be specific.
- probabilities are sorted descending, two decimals, sum to 1.00; if fewer than 3 candidates, pad with null/0.00.
- answer must equal probabilities[0].char and must be a single Chinese character.
- If uncertain, state it in rationale, but still provide the most likely answer.

Now output the JSON that fully follows these rules.
