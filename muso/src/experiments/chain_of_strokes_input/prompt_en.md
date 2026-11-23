PROMPT_TEMPLATE = """Task: Identify the Chinese character in a single calligraphy frame (this frame already uniquely determines the character).

Context:
- The image is a screenshot from a Chinese calligraphy teaching video.
- The character in the frame is unfinished; there are {x} strokes remaining.
- The model receives the current frame and text descriptions of all strokes completed so far.
- Prior stroke descriptions (in writing order):
{stroke_descriptions}

Output specification (strict):
- Output a single JSON object with no extra text, prefixes, suffixes, or Markdown.
- Fields:
  "answer": "<single Chinese character>",            // final decision (must match the top probability entry)
  "probabilities": [                                 // sorted by probability desc, length 3
    {"char":"<char1>","p":0.xx},
    {"char":"<char2>","p":0.xx},
    {"char":"<char3>","p":0.xx}
  ],
  "rationale": "<brief reasoning>"

Strict formatting:
- Probability p is a decimal between 0–1, two decimals; all p must sum to 1.00 (after rounding).
- If "answer" ≠ probabilities[0].char, set "answer" = probabilities[0].char.
- If fewer than 3 candidates exist, use null for the character and 0.00 for the probability.
- Only output Chinese characters as candidates; no words, pinyin, or symbols.

Calibration guidance (internal; still output only JSON):
- Assign relative scores s_i ∈ {{1..9}} to top-K candidates, then normalize to probabilities p_i = s_i / Σ s_i.
- If the top probability < 0.50, still provide the most likely answer.
- In "rationale", briefly explain the evidence and sources of ambiguity.

Now output the JSON that follows all rules above."""
