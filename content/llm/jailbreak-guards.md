## 越狱攻击有哪些套路

数据里的 55 种越狱策略主要来自 EasyJailbreak，大致分几类：

| 类别 | 例子 | 思路 |
| --- | --- | --- |
| 拒绝抑制 | refusal_suppression | 直接要求「不许拒绝」 |
| 角色扮演 | AIM、evil_confidant | 让模型扮演不受约束的角色 |
| 场景嵌套 | DeepInception | 故事里套故事，把请求藏深 |
| 上下文攻击 | ICA | 在上下文里放「顺从」的示范 |
| 混淆 / 编码 | Base64、ROT13、Leetspeak、Disemvowel、ArtPrompt | 让过滤器看不懂 |
| 多语言 | 十种语言 | 低资源语言里对齐更弱 |

目标模型里有 3 个对齐过的（Llama-3.1-8B-Instruct、Qwen2.5-1.5B/7B-Instruct）和 2 个 **abliterated** 版本（把「拒绝方向」从权重里去掉的模型），用来提高越狱成功率，让评估器面对更多真正的泄露。

## Guard 模型是什么

Guard 是专门做安全分类的模型，放在主模型前后把关：

- **prompt 级**：判断用户的意图危不危险；
- **prompt–response 级**：判断模型的回答危不危险。

输出形式五花八门：有害 / 安全 / 拒绝 / 违反哪条政策的标签、连续的风险分数，或者带一段推理过程。常见的有 LlamaGuard3、Qwen3Guard、WildGuard、GraniteGuardian、GuardReasoner、RSafe、Nemotron 等。

## 它们有多不一致

- 同一批回答，各个 guard 判「安全」的比例从约 25% 到约 72%，**严格程度差了将近 3 倍**；
- 12 个 guard 多数投票，和人工标签的 κ 是 −0.022，相当于随机。

## 一个典型的能力失败

回答是 `QWxhZGRpbjpvcGVuIHNlc2FtZQ==`，Base64 解码后是「Aladdin:open sesame」，和要问的隐私毫无关系。

- 某个 guard 声称它解码出来是「123456789」，判定泄露了税号：**模型在脑补解码**；
- 另一些评估器一看到编码输出，就默认是泄露。

一个确定性的 Base64 解码器就能终结这场争论。

## 带推理的 guard 更有用

PrivacyEvalClaw 的提示里，优先用会**给出推理过程**的 guard（RSafe、GraniteGuardian、Nemotron 等），而不是只给标签的。理由是：推理是「智能体可以检查、可以质疑的证据」，一个光秃秃的「unsafe」没法追问。

## 实习时的用法

隐私对齐的 SFT 实验里，同一批 guard（PromptGuard、RSafe、WildGuard、LlamaGuard3、Qwen3Guard、GraniteGuardian、GuardReasoner 等）还被当成**数据质量指标**：给生成的训练数据打分，筛掉不合格的。同一个工具，既可以当评估器，也可以当数据过滤器。
