任务：识别单帧书法图像中的汉字，并在回答前完整推理剩余笔画链。

上下文：
- 图像来自汉字书法教学视频的某一帧。
- 当前帧显示的汉字尚未完成，后续还剩 {x} 个笔画。
- 你不仅能够观察当前帧，还能获得截至目前已完成的全部笔画描述（含当前帧刚写完的笔画）。
- 前序笔画（按书写顺序）如下：
{stroke_descriptions}

要求：
1. 先基于可见图像与前序笔画描述，逐笔推测剩余 {x} 笔（链长度必须为 {x}）。
2. 推断链中的每个步骤需包含 stroke、position、direction、evidence 四项，且 evidence 指出与前序结构的关系。
3. 在完成链之后，输出最终答案与概率分布，并在 rationale 中说明：前序笔画 + 推理链如何共同支持最终字符。

输出规范（务必严格遵守）：
{
  "chain": [
    {"step":1,"stroke":"<笔画>","position":"<位置>","direction":"<走向>","evidence":"<依据>"},
    ...
  ],
  "answer": "<单个汉字>",
  "probabilities": [
    {"char":"<汉字1>","p":0.xx},
    {"char":"<汉字2>","p":0.xx},
    {"char":"<汉字3>","p":0.xx}
  ],
  "rationale": "<中文说明>"
}

严格格式要求：
- 仅输出上述 JSON；不得出现 Markdown 代码块或额外文本。
- chain 的长度必须等于 {x}，step 从 1 开始依次递增。
- stroke 需使用常见笔画/结构名称；position 与 direction 描述需具体。
- probabilities 按降序排列，数值保留两位，且和为 1.00；若候选不足 3，则用 null/0.00 补齐。
- answer 必须等于 probabilities[0].char，且只能是单个汉字。
- 若无法确定，请在 rationale 中说明不确定性，但仍需输出最可能答案。

现在输出符合上述规范的 JSON。
