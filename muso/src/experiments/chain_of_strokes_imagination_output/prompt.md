任务：识别当前书法帧中的汉字，并以“先想象前序、再预测后序”的链式方式推理。

上下文：
- 图像来自汉字书法教学视频的单帧截图。
- 当前帧之前的书写轨迹不会再额外提供，你只能依靠画面推断。
- 此时还剩 {x} 个笔画没有写完；在回答之前，必须先复原到当前帧为止的完整书写链，再推演这 {x} 个未来笔画。

推理步骤：
1. 仅凭可见结构，补全从第 1 笔到当前帧的全部笔画，形成 `imagined_prefix`（step 从 1 起连续递增，必须覆盖当前帧，不能跳笔或提前结束）。
2. 在此基础上，继续输出剩余的 `future_chain`，其长度必须等于 {x}。
3. 两段链的每个元素都要包含 stroke、position、direction、evidence 四项；evidence 需说明与已知结构或假设的关系。
4. 完成两段链后，再输出最终答案、概率分布与 rationale，说明“想象 + 预测”如何共同锁定该字。

输出格式（必须严格遵守）：
{
  "imagined_prefix": [
    {"step":1,"stroke":"<笔画>","position":"<位置>","direction":"<走向>","evidence":"<依据>"},
    {"step":2,"stroke":"<笔画>","position":"<位置>","direction":"<走向>","evidence":"<依据>"}
    ...
  ],
  "future_chain": [
    {"step":1,"stroke":"<笔画>","position":"<位置>","direction":"<走向>","evidence":"<依据>"},
    ...
    {"step":{x},"stroke":"<笔画>","position":"<位置>","direction":"<走向>","evidence":"<依据>"}
  ],
  "answer": "<单个汉字>",
  "probabilities": [
    {"char":"<汉字1>","p":0.xx},
    {"char":"<汉字2>","p":0.xx},
    {"char":"<汉字3>","p":0.xx}
  ],
  "rationale": "<说明：当前帧可见结构 + 想象前序链 + 预测后序链 如何共同锁定该字>"
}

输出要求：
- 仅输出上述 JSON，无额外文本、解释、Markdown 或注释。
- `imagined_prefix` 必须从第 1 步开始连续覆盖至当前帧；`future_chain` 的长度必须等于 {x}。
- stroke 使用常见笔画或结构名称；position/ direction 要具体；evidence 用短句说明依据或不确定性来源。
- 概率 p 需按降序排列，保留两位小数，且和为 1.00；候选不足 3 时用 null/0.00 补齐。
- answer 必须等于 probabilities[0].char，且只能是单个汉字。
- rationale 应点明关键部件、哪些笔画区分候选、想象链与预测链如何闭合整字。

校准提示（内部使用）：
- 想象前序链需与当前帧可见结构一致，不得出现与画面矛盾的笔画或部件。
- future_chain 要继承 imagined_prefix 的结构，避免重复或违背部件书写顺序。
- 当存在不确定时，可在 evidence 中指出模糊点，但仍需给出最可能的完整链与答案。
- 采用评分归一化方式分配概率，确保最高候选也不过度自信（除非结构极为确定）。

请输出满足上述所有要求的 JSON。
