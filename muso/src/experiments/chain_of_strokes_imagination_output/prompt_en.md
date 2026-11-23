Task: Identify the character in the current calligraphy frame and reason in a two-part chain: first imagine the prefix, then predict the suffix.

Context:
- The image is a single-frame screenshot from a Chinese calligraphy teaching video.
- No extra info about prior strokes beyond the visible frame will be given; you must infer from the image alone.
- There are {x} strokes left to write; before answering, reconstruct the full writing chain up to the current frame, then predict those {x} future strokes.

Reasoning steps:
1. Using only the visible structure, complete all strokes from the first stroke up to the current frame to form `imagined_prefix` (step starts at 1, continuous, must cover the current frame; no skipping or early stopping).
2. On that basis, output the remaining `future_chain`, whose length must equal {x}.
3. Every element in both chains must include stroke, position, direction, and evidence; evidence should state the relation to known structure or assumptions.
4. After both chains, output the final answer, probability distribution, and rationale explaining how “imagination + prediction” jointly pin down the character.

Output format (must strictly follow):
{
  "imagined_prefix": [
    {"step":1,"stroke":"<stroke>","position":"<position>","direction":"<direction>","evidence":"<evidence>"},
    {"step":2,"stroke":"<stroke>","position":"<position>","direction":"<direction>","evidence":"<evidence>"},
    ...
  ],
  "future_chain": [
    {"step":1,"stroke":"<stroke>","position":"<position>","direction":"<direction>","evidence":"<evidence>"},
    ...
    {"step":{x},"stroke":"<stroke>","position":"<position>","direction":"<direction>","evidence":"<evidence>"}
  ],
  "answer": "<single Chinese character>",
  "probabilities": [
    {"char":"<char1>","p":0.xx},
    {"char":"<char2>","p":0.xx},
    {"char":"<char3>","p":0.xx}
  ],
  "rationale": "<explain how visible structure + imagined prefix + predicted suffix jointly lock in the character>"
}

Output requirements:
- Output only the JSON above—no extra text, explanations, Markdown, or comments.
- `imagined_prefix` must start at step 1 and continuously cover the current frame; `future_chain` length must equal {x}.
- stroke uses common stroke/structure names; position/direction must be specific; evidence uses short sentences to state support or uncertainty.
- Probabilities are in descending order, two decimals, sum to 1.00; if fewer than 3 candidates, pad with null/0.00.
- answer must equal probabilities[0].char and be a single Chinese character.
- rationale should highlight key components, discriminative strokes, and how the imagined/predicted chains close the character.

Calibration tips (internal):
- The imagined prefix must match the visible structure; do not invent strokes that conflict with the image.
- future_chain must inherit the imagined_prefix structure, avoiding repetition or order conflicts.
- If uncertain, note ambiguities in evidence but still provide the most probable full chain and answer.
- Use score normalization to assign probabilities, avoiding overconfidence unless the structure is very certain.

Please output the JSON that satisfies all requirements above.
