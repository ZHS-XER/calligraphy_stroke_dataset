Task: Identify the target Chinese character from a sequence of calligraphy frame images where the final frame already contains enough information to uniquely determine the character.

- The images come from a Chinese calligraphy teaching video and show the stroke-by-stroke writing of the same character.
- You receive several preceding frames sampled in chronological order at roughly equal intervals.
- The last frame (latest in the chain) contains all the key information needed to decide the character.
- All frames are stitched into one long image from left to right; that long image is what you are given.
- In the current (last) frame there are {x} strokes remaining to be written.

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
- Probabilities p are decimals between 0–1, two decimal places; they must sum to 1.00 (after rounding).
- If "answer" ≠ probabilities[0].char, set "answer" = probabilities[0].char.
- If fewer than 3 candidates exist, use null for the character and 0.00 for the probability.
- Only output Chinese characters as candidates; no words, pinyin, or symbols.

Calibration guidance (internal, still output only JSON):
- First assign relative scores s_i ∈ {1..9} to top-K candidates, then normalize to probabilities p_i = s_i / Σ s_i.
- Even if the top probability < 0.50, still provide the most likely answer.
- In "rationale", briefly state the evidence and which earlier-frame cues help confirm the character (e.g., radical shape, structure completion, stroke trends).

Now output the JSON that follows all rules above.
