Task: Identify the Chinese character in a single calligraphy frame (this frame already contains enough information to uniquely determine the character).

Context:
- The image is from a Chinese calligraphy teaching video.
- The current frame contains key strokes/structure sufficient to judge the target character.
- There are {x} strokes remaining; before giving the answer, you must stepwise predict these {x} remaining strokes.

Output specification (strict):
- Output only one JSON object with no extra text, prefixes, or comments.
- Fields:
  "chain": [                                      // language-based stepwise prediction chain; length must be {x}
    {"step":1,"stroke":"<stroke name>",           // e.g., "horizontal" / "vertical" / "left-falling" / "horizontal-turn" / "hook"
     "position":"<relative position>",            // e.g., "upper-left / middle / lower-right / slightly left / slightly upper / right of center axis"
     "direction":"<trajectory or structural relation>", // e.g., "left-to-right with slight rise" / "top-to-bottom through center axis" / "joins left vertical to form bottom of a '口'"
     "evidence":"<evidence>"
    },
    {"step":2,"stroke":"<stroke name>",
     "position":"<relative position>",
     "direction":"<trajectory or structural relation>",
     "evidence":"<brief evidence>"
    },
    // ... until step = {x}
  ],
  "answer": "<single Chinese character>",         // final decision (must equal probabilities[0].char)
  "probabilities": [                              // sorted by probability desc, length 3
    {"char":"<char1>","p":0.xx},
    {"char":"<char2>","p":0.xx},
    {"char":"<char3>","p":0.xx}
  ],
  "rationale": "<summary: visible frame + stepwise chain -> why this character>"
}

Strict formatting requirements:
- The "chain" length must equal {x}; step increases from 1 with no gaps or repeats.
- Each step must include stroke, position, direction, evidence; all are short phrases with specific details. If uncertain, note insufficient evidence in evidence.
- Probabilities p are decimals between 0–1, two decimals; all p must sum to 1.00 (after rounding).
- If "answer" ≠ probabilities[0].char, set "answer" = probabilities[0].char.
- If fewer than 3 possible characters, use null for the missing characters and 0.00 for probabilities.
- Only output single Chinese characters as candidates; do not output words, pinyin, or symbols.
- Output nothing except the JSON (no explanations, prefixes/suffixes, or Markdown markers).

Calibration guidance (internal; only JSON should be output):
- Ensure structural consistency in the chain: later strokes must not contradict earlier predicted structure (e.g., don't add an inward horizontal after closing a "口").
- Prefer strokes that close components or complete key structures (e.g., "口", "日", "门"); if using component priors, mention the component in evidence.
- First assign relative scores s_i ∈ {1..9} to top-K candidates, then normalize to p_i = s_i / Σ s_i; even if top p < 0.50, still output the most likely answer.
- In "rationale", summarize which steps are decisive and how the language chain plus visible strokes reinforce each other.

Now output the JSON that satisfies all requirements above.
