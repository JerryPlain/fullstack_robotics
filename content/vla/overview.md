> [!one] VLA（Vision-Language-Action Model，视觉-语言-动作模型）= 在互联网图文上预训练过的 VLM，再用机器人示范教它输出动作。它学的是 $\pi_\theta(A_t \mid \text{images}, \text{instruction}, q_t)$。押注的是：web 上学到的"这是什么、在哪、指令什么意思"能迁移成语义层面的泛化，而"怎么动"仍然要从机器人数据里学。

## 直觉

传统的视觉运动策略（比如 [Diffusion Policy](#/learn/manip/diffusion-policy)）通常一个任务训一个模型。它从几百条示范里学会"把红方块推到目标区"，但你说"把那个长得像恐龙的东西放进写着 3 的碗里"，它就完全没办法了：训练数据里既没有恐龙，也没有数字 3。

[VLM](#/learn/found/vlm) 恰好擅长这些：它见过几十亿张图文对，认识恐龙，会读数字，能理解"左边那个更大的"。**VLA 的想法是把这个"读过万卷书、但从没有过身体"的大脑接到机器人身上**，再用机器人数据教它运动技能。具体有三个期待：

1. **语义泛化**：没在机器人数据里见过的物体、背景和说法，也能处理；
2. **语言作为通用任务接口**：一个模型、无数任务，换任务只需要换指令；
3. **Scaling**：像 LLM 一样，数据和模型规模上去之后能力持续提升，最终得到一个通用的机器人基础模型。

也要清楚它的边界：VLM 知道"是什么"和"在哪"，但不知道"要用多大力、手腕怎么转、接触时会发生什么"。**这部分物理和运动知识，web 数据几乎给不了**，只能靠机器人数据。这也是为什么 VLA 至今仍然受数据瓶颈限制。

## 形式化

把一次决策写成：输入 $n$ 路相机图像 $I_t^{1:n}$、语言指令 $\ell$、本体状态 $q_t$（关节角、末端位姿、夹爪开度），输出一段动作 chunk $A_t = [a_t, \dots, a_{t+H-1}]$：

$$
A_t \sim \pi_\theta\big(A_t \mid I_t^{1:n},\ \ell,\ q_t\big)
$$

模型 = 视觉编码器 + LLM backbone + **动作输出方式**。训练目标取决于动作怎么输出。

**离散 token（自回归）**：先把每个连续动作维度分桶，比如 256 个 bin：

$$
b = \Big\lfloor \frac{a - a_{\min}}{a_{\max} - a_{\min}} \cdot 255 \Big\rceil
$$

然后和语言一样做 next-token prediction：

$$
\mathcal{L}_{\text{AR}} = -\sum_{j} \log p_\theta\big(\text{tok}_j \mid \text{tok}_{<j},\ I,\ \ell\big)
$$

**连续生成头（flow matching / diffusion）**：VLM 提供条件特征，action expert 回归速度场。$A^\tau = \tau A + (1-\tau)\epsilon$，$\epsilon \sim \mathcal{N}(0, I)$：

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{\tau,\,\epsilon}\Big[\big\| v_\theta(A^\tau, o, \tau) - (A - \epsilon) \big\|^2\Big]
$$

**Co-training**：机器人数据和 web 视觉-语言数据混在一起训练，防止 VLM 在机器人数据上把"常识"忘光：

$$
\mathcal{L} = \lambda\,\mathcal{L}_{\text{robot}} + (1-\lambda)\,\mathcal{L}_{\text{web}}
$$

## 发展脉络

| 年份 | 模型 | 骨干 / 规模 | 动作输出 | 数据 | 关键点 |
| --- | --- | --- | --- | --- | --- |
| 2022 | [RT-1](#/learn/vla/rt) | EfficientNet + FiLM 语言条件 + TokenLearner + Transformer，约 35M | 每维 256 bin 离散 | 约 13 万条真机示范，700+ 任务，13 台机器人 | 证明"大规模真机数据 + transformer"能扩展；还不是 VLA（没有 VLM 预训练） |
| 2023 | [RT-2](#/learn/vla/rt) | PaLI-X（5B/55B）、PaLM-E（12B） | **动作写成文本 token**：8 个整数，256 bin | RT-1 数据 + web 数据 co-fine-tune | 涌现出符号理解、简单推理等能力；55B 版本在云端 TPU 上约 1–3 Hz |
| 2023 | Open X-Embodiment / RT-X | RT-1-X、RT-2-X | 同上 | 22 种机器人，21 家机构，100 万+ 条轨迹 | 跨本体（cross-embodiment）数据能带来正迁移 |
| 2024 | Octo | Transformer，27M / 93M，T5 语言编码器 | **diffusion head** | OXE 中 80 万条轨迹 | 开源；可以微调到新的观测和动作空间；严格说没有 VLM 骨干 |
| 2024 | [OpenVLA](#/learn/vla/openvla) | Prismatic-7B：DINOv2 + SigLIP 视觉，Llama 2 7B | 256 bin（按 1%–99% 分位数），覆盖 Llama 词表里最少用的 256 个 token | OXE 中约 97 万条 episode；64×A100 训练 14 天 | 开源 7B，参数少 7 倍却比 RT-2-X 高 16.5 个百分点；支持 LoRA |
| 2024 | [π0](#/learn/vla/pi0) | PaliGemma 3B + 约 300M action expert | **flow matching**，chunk H = 50 | 约 1 万小时机器人数据：主体是自采（7 种机器人构型、68 个任务），另混入 OXE 等开源数据 | 叠衣服这类灵巧长程任务；最高 50 Hz 控制 |
| 2025 | [FAST](#/learn/vla/tokenize) / π0-FAST | π0 骨干，自回归 | **DCT + BPE 压缩 token** | 通用版 FAST+ 在约 100 万段动作 chunk 上训练 | 训练比扩散版 π0 快最多 5 倍；自回归推理更慢 |
| 2025 | π0.5 | 基于 π0 | 先预测文本子任务，再由 flow expert 出动作 | 多机器人、约 100 个家庭的移动操作数据、web 数据、子任务标注、口头指令 | 在完全没见过的新家里完成收拾厨房、卧室这类长程任务 |
| 2025 | GR00T N1（NVIDIA） | Eagle-2 VLM（System 2，约 10 Hz）+ DiT（System 1，约 120 Hz），约 2.2B | flow matching，chunk H = 16 | "数据金字塔"：web 与人类视频、合成数据、真机遥操作 | 开放的人形机器人基础模型；无动作视频用 latent action 标注 |
| 2025 | Gemini Robotics（Google DeepMind） | Gemini 2.0；云端 backbone + 机载 local action decoder | 动作 chunk，端到端延迟约 250 ms，等效 50 Hz | 数千小时 ALOHA 2 遥操作数据 + web/多模态数据 | 强泛化和灵巧性；配套 Gemini Robotics-ER 负责具身推理 |

这条线索大致是：**RT-1 证明数据规模有用 → RT-2 证明 VLM 知识能迁移 → OXE 证明跨本体数据能聚起来 → OpenVLA/Octo 把它开源 → π0 把动作头换成连续的 flow matching，解决了灵巧和高频控制 → π0.5、GR00T、Gemini Robotics 走向开放世界、人形机器人和分层推理。**

## 方法：动作怎么从语言模型里出来

### 三类动作输出

| 家族 | 代表 | 机制 | 优点 | 缺点 |
| --- | --- | --- | --- | --- |
| 离散 token（逐维分桶） | RT-2、OpenVLA | 每个维度、每个时间步一个 token，自回归解码 | 完全复用 LLM 的训练和推理基础设施；动作和语言共用一个 loss，co-training 很自然 | 精度受 bin 宽限制；高频 chunk 下 token 数爆炸，而且相邻 token 高度相关；推理慢 |
| 连续生成头（diffusion / flow） | Octo、π0、GR00T N1 | VLM 出条件特征，独立的 action expert 迭代去噪出整段连续 chunk | 连续、高精度、能表达多峰；一次出 50 步，适合 50 Hz 灵巧控制 | 多一个模块；flow loss 和语言 loss 不统一，action expert 的梯度可能破坏 VLM 原有的知识 |
| 压缩 tokenization | π0-FAST；π0.5 预训练阶段 | 对 chunk 做 DCT → 量化 → BPE，得到少量信息密集的 token | 保留 next-token 训练的效率和统一性；训练快 | 仍然自回归，推理慢（论文报告每个 chunk 约 750 ms，扩散版 π0 约 100 ms）；可能解码出非法序列 |

还有一条务实路线：**OpenVLA-OFT** 直接用 L1 回归连续 chunk 加并行解码，在 LIBERO 上把平均成功率从 76.5% 提到 97.1%，吞吐提升约 26 倍。它说明在示范比较单峰的场景里，"能不能多峰"没有"快不快、chunk 有没有做好"重要。

**为什么逐维分桶在高频下会失效**（FAST 论文的论点）：50 Hz 控制时，相邻时刻的动作几乎相同，token 序列高度冗余，模型只要"复制上一个 token"就能把 loss 压得很低，学不到有用的东西。DCT 把一段轨迹变成频域系数，大部分能量集中在少数低频系数上，去掉冗余之后每个 token 都携带信息。

π0.5 把两者结合起来：**预训练用 FAST 离散 token**（训练快，和 web 数据统一），**后训练加上 flow matching action expert**（推理快、动作连续），推理时用 10 步去噪。

### 双系统与分层

一个 3B 的 VLM 很难以 50 Hz 跑完整前向，而且语义推理本来就不需要这么高的频率。于是很多系统借用"System 1 / System 2"的说法（详见[分层 VLA](#/learn/vla/hierarchy)）：

- **System 2（慢）**：大 VLM，理解场景和指令、拆解子任务，几 Hz 到 10 Hz；
- **System 1（快）**：小的 action expert 或策略网络，根据 System 2 的 latent 或子任务和最新观测出动作，几十到上百 Hz。

GR00T N1 是 10 Hz / 120 Hz；Gemini Robotics 用云端 backbone 加机载 decoder 来补偿延迟；π0.5 则在同一个模型里先用文本预测子任务（"拿起盘子"），再出低层动作。共同的直觉：**语义变化慢、物理变化快，所以分开算**。

### 数据从哪里来

（详见 [VLA 数据](#/learn/vla/data)。）

- **真机遥操作示范**：质量最高也最贵。π0 预训练的约 1 万小时数据以自采遥操作为主，Gemini Robotics 也用了数千小时遥操作数据（采集方式见[遥操作](#/learn/manip/teleop)）；
- **跨本体数据**：OXE 把几十家实验室的数据统一格式。动作空间、控制频率、相机位置各不相同，需要按本体归一化；
- **web 数据 co-training**：RT-2 发现，和 web 数据一起训比只用机器人数据微调泛化更好（PaLI-X 版本里机器人数据约占一半）；
- **人类视频 / 合成数据**：没有动作标签，就用 latent action 或逆动力学模型（inverse dynamics model, IDM）补标，GR00T N1 就是这么做的；也可以用仿真或[视频世界模型](#/learn/wm/video)生成数据。

### 还没解决的问题

- **数据稀缺**：最大的机器人数据集也只有几千到几万小时，和 LLM 的万亿级 token 差好几个数量级。而且动作数据绑定本体，不能简单相加。
- **评测**：真机评测慢、贵、难以复现；仿真 benchmark 又容易饱和，和真机表现的相关性也存疑。领域里缺少公认的、可复现的通用评测。
- **精度与灵巧性**：多数 VLA 只用视觉和本体状态，没有力觉和触觉，毫米级插装、力控类任务仍然很难。
- **RL 微调**：模仿学习的上限是示范质量。怎么用真机 RL 让 VLA 超越示范、从自己的失败里学习，是很活跃的方向（见 [RL 微调](#/learn/rl/finetune)）。
- **记忆与长程任务**：多数 VLA 只看当前一两帧，"我刚才把钥匙放哪了"这类需要记忆的任务基本做不了。分层推理部分缓解了这个问题，但没有根本解决。

## 机器人视角

> [!robot] **推理延迟是第一约束**：RT-2-55B 约 1–3 Hz（云端），OpenVLA 7B 在 RTX 4090 上约 6 Hz，π0 在 4090 上出一个 50 步 chunk 约 73 ms（机载）。所以 action chunking 几乎是必需的：一次推理出一段动作，执行的同时异步推下一段。还要处理 chunk 之间的衔接，比如时间戳过期的动作丢掉、chunk 交界处做平滑。

> [!robot] **微调才是常态**：拿到预训练 VLA 后，换新机器人、新相机或新任务基本都要微调（见 [VLA 微调](#/learn/vla/finetune)）。OpenVLA 报告 rank 32 的 LoRA 只训约 1.4% 的参数，就能达到全量微调的效果；4-bit 量化后约 7 GB 显存，成功率基本不掉。开源起点可以用 openpi（π0 / π0.5）和 [LeRobot](https://github.com/huggingface/lerobot)。

> [!robot] **动作空间和归一化统计量要对齐**：预训练数据里有末端增量、绝对关节角、不同的夹爪约定。OpenVLA 推理时需要指定用哪个数据集的统计量来反归一化，选错了动作尺度就全错。自己的机器人最好在微调时统一用一种动作表示，并固定控制频率。

> [!robot] **相机摆位是隐藏的分布偏移**：VLA 对"语义"泛化得不错，对"视角"泛化很差。部署时的相机位置、手腕相机有没有、分辨率，尽量和微调数据保持一致。

## 常见坑

> [!pitfall] **以为 VLA 懂物理**：VLA 的泛化主要是语义层面的（新物体、新说法），不是运动层面的（新的抓取方式、新的接触动力学）。换了夹爪或者物体重量变了，它不会"想明白"，需要相应的数据。

> [!pitfall] **微调把 VLM 训傻**：只用少量机器人数据全量微调，模型很快过拟合，语言理解和泛化能力退化（灾难性遗忘）。缓解办法：LoRA、冻结部分层、保留一部分 web 数据做 co-training，或者把 action expert 和 VLM 的梯度隔离开。

> [!pitfall] **在饱和的 benchmark 上比高低**：LIBERO 这类仿真 benchmark 已经被刷到 95% 以上，几个点的差距说明不了真机表现。关心泛化的话，要在真机上用没见过的物体、场景和指令做足够多次试验（见[评测](#/learn/manip/eval)）。

## 自测

> [!quiz] RT-2 为什么能把动作直接当成文本 token 来训练？这样做的代价是什么？
> 动作每维分成 256 个 bin，一个动作变成 8 个整数组成的字符串，就和 VQA 的回答格式一样了，可以直接用 VLM 的 next-token loss，和 web 数据混在一起训练，不改架构。代价有：（1）精度受 bin 宽限制；（2）每个动作要自回归解码多个 token，大模型下推理只有 1–3 Hz；（3）每一步只出一个动作，没有 chunk，难以支撑高频灵巧控制；（4）token 之间的数值关系（bin 100 和 101 很接近）要模型自己学。

> [!quiz] 为什么逐维分桶在 50 Hz、50 步 chunk 的设定下效果很差？FAST 怎么解决？
> 一个 chunk 是 50 步 × 若干维，变成几百个 token，序列长、推理慢；更要命的是相邻时间步几乎一样，next-token 目标可以靠"复制上一个 token"轻松降低，学习信号很弱。FAST 先对每个维度做 DCT，把轨迹转到频域，能量集中在低频；再按比例缩放并取整，丢掉不重要的系数；按"低频优先"的顺序展平后用 BPE 压缩。最终每个 chunk 只有几十个信息密集的 token。

> [!quiz] 双系统架构为什么要分快慢？如果直接让一个 3B 的 VLM 以 50 Hz 逐步输出动作会怎样？
> 3B 模型一次前向就要几十毫秒，再加上视觉编码，很难稳定跑到 50 Hz，更不用说自回归出多个 token。语义（任务是什么、目标在哪）变化慢，几 Hz 更新就够；而接触、平衡这类物理反馈要求高频闭环。分开之后，大模型低频地提供语义 latent 或子任务，小模型高频地出动作，两者各取所需。代价是两部分的接口（传 latent 还是传文本、两者是否联合训练）需要精心设计。

> [!quiz] Co-training 时加入 web 数据有什么作用？什么情况下可以不加？
> 作用是在学动作的同时保住 VLM 的语义知识，否则机器人数据量少、分布窄，模型会遗忘，失去对新物体和新指令的泛化。RT-2 的实验显示 co-fine-tune 比只用机器人数据微调泛化更好。如果是下游的单任务微调（固定场景、固定指令），只追求这个任务的成功率，那么不加 web 数据、只用 LoRA 微调通常也可以，但开放世界的泛化会变差。

> [!quiz] Octo 算不算 VLA？这个问题说明了什么？
> 按严格定义（基于互联网预训练的 VLM），Octo 不算：它的 transformer 是在 OXE 上从头训练的，只用 T5 编码语言，没有继承 VLM 的视觉-语言知识。但它和 VLA 一样以语言或目标图像为条件，在跨本体数据上预训练，然后微调。这说明"VLA"更多是一个连续谱系：一端是纯机器人数据训练的通用策略，另一端是 VLM 加动作头。真正决定泛化的是预训练数据里有没有 web 知识、动作头的设计，以及机器人数据的规模和多样性。

## 延伸阅读

- [RT-2（arXiv:2307.15818）](https://arxiv.org/abs/2307.15818) — "动作即文本"的开端，涌现能力那一节值得看
- [Open X-Embodiment（arXiv:2310.08864）](https://arxiv.org/abs/2310.08864) — 跨本体数据集和 RT-X
- [Octo（arXiv:2405.12213）](https://arxiv.org/abs/2405.12213) — 开源通用策略，diffusion head 的设计
- [OpenVLA（arXiv:2406.09246）](https://arxiv.org/abs/2406.09246) — 开源 7B VLA，LoRA 和量化实验很实用
- [π0（arXiv:2410.24164）](https://arxiv.org/abs/2410.24164) — flow matching action expert 和 attention mask 的设计
- [FAST（arXiv:2501.09747）](https://arxiv.org/abs/2501.09747) — 为什么动作 tokenization 很重要
- [OpenVLA-OFT（arXiv:2502.19645）](https://arxiv.org/abs/2502.19645) — 微调 VLA 的工程配方：chunking、并行解码、连续动作
- [π0.5（arXiv:2504.16054）](https://arxiv.org/abs/2504.16054) — 异构数据 co-training 和分层推理
- [GR00T N1（arXiv:2503.14734）](https://arxiv.org/abs/2503.14734) — 人形机器人的双系统和数据金字塔
- [Gemini Robotics（arXiv:2503.20020）](https://arxiv.org/abs/2503.20020) — 云端加机载的部署架构
- [openpi](https://github.com/Physical-Intelligence/openpi) — π0 / π0-FAST / π0.5 的官方开源权重和微调代码
- [LeRobot](https://github.com/huggingface/lerobot) — 数据格式和多种策略的统一实现

> [!mine] 我的笔记：学完后在这里写下自己的理解、疑问和实验记录。
