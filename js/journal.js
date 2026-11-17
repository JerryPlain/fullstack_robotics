// ============================================================
// 学习记录 —— 整个网站的数据，手动维护
// ------------------------------------------------------------
// 一个单位是「项目」。做完一个项目：
//   1. PROJECTS 里加一条；content/projects/<id>.md 写项目总结
//   2. 学到的每个知识点写成 content/<领域>/<slug>.md，并在 NOTES 里登记
//      （from = 来自哪个项目；remember = 以后要记住的基础，一条一句）
//   3. 用到了以前的笔记，就把它们的 id 写进项目的 uses
// 路线图、项目页、知识页、搜索都从这里自动生成。
// ============================================================

// 层：整个技术栈从上（应用）到下（地基）。颜色跟着层走。
const LAYERS = [
  { id: "domain", t: "领域", en: "Domains" },
  { id: "intel",  t: "智能", en: "Intelligence" },
  { id: "body",   t: "身体", en: "Body" },
  { id: "base",   t: "地基", en: "Foundations" },
];

// 领域：顺序 = 路线图上从上到下的顺序。需要新领域就加一行（写清在哪一层）。
const SECTIONS = [
  { id: "med",    layer: "domain", t: "医疗机器人", en: "Medical Robotics" },

  { id: "agent",  layer: "intel",  t: "智能体",     en: "Agents" },
  { id: "llm",    layer: "intel",  t: "大模型",     en: "LLMs & MLLMs" },
  { id: "vla",    layer: "intel",  t: "VLA",        en: "Vision-Language-Action" },
  { id: "wm",     layer: "intel",  t: "世界模型",   en: "World Models" },
  { id: "learn",  layer: "intel",  t: "机器人学习", en: "Robot Learning" },
  { id: "perc",   layer: "intel",  t: "感知与表征", en: "Perception & Representation" },

  { id: "manip",  layer: "body",   t: "操作",       en: "Manipulation" },
  { id: "loco",   layer: "body",   t: "运动",       en: "Locomotion" },
  { id: "sim",    layer: "body",   t: "仿真",       en: "Simulation" },
  { id: "ctrl",   layer: "body",   t: "控制与规划", en: "Control & Planning" },
  { id: "hw",     layer: "body",   t: "硬件与传感", en: "Hardware & Sensing" },

  { id: "infra",  layer: "base",   t: "工程与算力", en: "Engineering & Compute" },
  { id: "method", layer: "base",   t: "科研方法",   en: "Research Methods" },
  { id: "math",   layer: "base",   t: "数学基础",   en: "Math & ML Foundations" },
];

// 项目：按 date 排序（可以补录更早的项目，路线图会自己排好）。
//   t 短标题（路线图上显示）  sub 副标题  full 完整题目
//   one 一句话  uses 用到的以前项目的笔记 id
const PROJECTS = [
  {
    id: "swarmagentic",
    date: "2025-06",
    t: "SwarmAgentic",
    sub: "用粒子群在语言空间里自动设计多智能体系统",
    full: "SwarmAgentic: Towards Fully Automated Agentic System Generation via Swarm Intelligence",
    one: "只给任务描述和目标函数，从零生成多智能体系统，再用 PSO 演化一群候选系统：差（diff）和改（apply）都交给 LLM 在文本里完成。",
    uses: [],
  },
  {
    id: "privacyevalclaw",
    date: "2026-04",
    t: "PrivacyEvalClaw",
    sub: "审计大模型隐私泄露的审议式智能体",
    full: "PrivacyEvalClaw: A Tool-Augmented Deliberative Agent for Auditing Privacy Leakage in LLM Jailbreak Responses",
    one: "guard 和 LLM judge 对「这算不算泄露」经常吵架；把它们都当成工具、把分歧当成要调查的问题，一个不训练的智能体就比最好的单个 judge 更接近人的判断。",
    uses: [],
  },
  {
    id: "surgagentgym",
    date: "2026-05",
    t: "SurgAgentGym",
    sub: "通用多模态大模型和工具智能体的外科场景理解基准",
    full: "How to “Train” a Surgeon? Benchmarking Generalist Agents in Surgical Scene Understanding",
    one: "先摸底通用 MLLM 懂多少外科，再给它们分割工具和外科知识库：工具帮了感知类任务，但失败的智能体和成功的一样爱用工具，差距在于怎么用证据。",
    uses: ["method/agreement-kappa", "infra/vllm-inference"],
  },
  {
    id: "representative-atlas",
    date: "2026-11",
    t: "Representative Atlas",
    sub: "机器人超声导航：图谱检索 vs 世界模型",
    full: "Representative Atlas or World Models for Robotic Ultrasound Navigation",
    one: "在没见过的病人身上，只用当前一张超声图把探头开到 L4：先估计「我在解剖上的哪里」，比预测「下一帧长什么样」停得更准，也便宜得多。",
    uses: [],
  },
];

// 笔记：id = "<领域>/<slug>"，正文在 content/<领域>/<slug>.md
// 同一领域里按这里的顺序排。t 尽量短（路线图上显示）。
const NOTES = [
  // ---- Representative Atlas ----
  {
    id: "med/robotic-ultrasound", from: "representative-atlas",
    t: "机器人超声", en: "Robotic Ultrasound",
    remember: [
      "超声只给一个局部切面，长什么样取决于探头的位姿和压力。机器人超声的核心难题是**知道自己在解剖上的哪里**。",
      "RL / IL 学的是「观测 → 动作」的直接映射，解剖知识藏在策略里：看不出它认为自己在哪，换目标就要重训。",
      "闭环形式（SonoGym）：输入当前 B-mode 图 + 里程计，输出体表上 2 个平移 + 绕法向 1 个旋转。",
    ],
  },
  {
    id: "med/ultrasound-physics", from: "representative-atlas",
    t: "超声成像原理", en: "Ultrasound Physics",
    remember: [
      "软组织声速按 1540 m/s 算，深度 $d = ct/2$（声波走了一个来回）。",
      "界面反射 $R = \\left(\\frac{Z_2-Z_1}{Z_2+Z_1}\\right)^2$，$Z=\\rho c$。软组织→骨约 0.4，软组织→空气约 1（所以要耦合剂）。",
      "骨面是一条亮线，后面是声影：超声只看得到骨的**表面**。",
      "频率越高分辨率越好、穿透越浅；衰减约 0.5 dB/(cm·MHz)，按来回路程算。",
      "Speckle 是分辨单元里大量散射体的干涉图样：位姿不变它就不变，既不是随机噪声，也不是解剖结构。",
    ],
  },
  {
    id: "med/lumbar-spine", from: "representative-atlas",
    t: "腰椎解剖与定位", en: "Lumbar Anatomy & Level Finding",
    remember: [
      "L1–L5 相邻椎体中心相距约 30–34 mm。从背后能看到的是后部结构：棘突、椎板、关节突、横突。",
      "相邻腰椎在 B-mode 里很像，同一节在体表的位置又因人而异，所以单看一张图很难说出这是第几节。",
      "临床定位：Tuffier 线（两侧髂嵴最高点连线）一般过 L4 棘突或 L4/5 间隙，但个体差异大。超声上常从骶骨往上数。",
    ],
  },
  {
    id: "sim/ultrasound-rendering", from: "representative-atlas",
    t: "从 CT 渲染超声", en: "Rendering Ultrasound from CT",
    remember: [
      "导航先要几何对（骨面在对的深度），其次才是看起来像。PSNR / SSIM 高不代表能用。",
      "MB+CT：组织标签决定声学参数，CT 强度修正反射项。只用标签渲染，边界太硬，骨后面还能「看见」结构。",
      "固定的 speckle 会和病人坐标绑在一起，等于给策略留了位置捷径，所以每帧重新采样散射体。",
      "gCNR = 1 − 两个区域灰度分布的重叠，取值 0–1，对灰度映射不敏感。",
    ],
  },
  {
    id: "perc/anatomical-atlas", from: "representative-atlas",
    t: "解剖图谱与规范坐标", en: "Anatomical Atlas & Canonical Coordinates",
    remember: [
      "规范坐标系要先量化验证：对应结构跨病人的离散度（SD）要小，相邻结构的重叠（soft Jaccard）要低。",
      "纵轴沿 L5→L1 连线定义，再减掉一条二次曲线把腰椎前凸拉直。只变换坐标、不重采样图像；这是个剪切，闭式可逆。",
      "验证对齐时留出不参与拟合的结构（只用 L1/L3/L5 拟合，看 L4），这样对齐好坏是测出来的，不是拟合出来的。",
      "人群先验有天花板：人群 L4 目标横向 SD 19.7 mm，比纵向 15.2 mm 还大。",
    ],
  },
  {
    id: "perc/metric-retrieval", from: "representative-atlas",
    t: "度量学习与检索定位", en: "Metric Learning & Retrieval Localization",
    remember: [
      "让 embedding 距离直接逼近物理距离：$\\ell_{ij} = \\big(\\lVert e_i-e_j\\rVert - D(u_i,u_j)\\big)^2$。比对比学习多了「单位」：距离就是 mm。",
      "检索到的 k 个邻居当成多个位置假设，**不要取平均**：L3 和 L5 的平均正好是 L4。",
      "打乱 bank 里的坐标后，最后一帧有 L4 的比例从 73% 掉到 8%：检索的价值全在坐标上。",
    ],
  },
  {
    id: "perc/visual-encoders", from: "representative-atlas",
    t: "DINOv2 与 V-JEPA 2", en: "Frozen Visual Encoders",
    remember: [
      "DINOv2：图像上的自蒸馏。V-JEPA 2：在表征空间里预测被遮住的视频块。两者都是在自然图像/视频上训的。",
      "冻结特征之间的距离衡量的是「看起来像」，不是「解剖上近」。",
    ],
  },
  {
    id: "wm/latent-planning", from: "representative-atlas",
    t: "潜空间世界模型规划", en: "Planning with Latent World Models",
    remember: [
      "潜空间世界模型 = 冻结编码器 $E$ + 动作条件预测器 $\\hat z_{t+1} = P(z_t, a)$。规划时比较预测特征和目标特征，不生成图像。",
      "没有目标图时，拿训练集里目标 15 mm 以内的图组成目标集合，代价 = 到最近 5 个目标特征的平均距离。",
      "能走到附近 ≠ 能停下。外观代价在目标附近量的不是「离目标多远」：世界模型进入 30 mm 的 episode 里有 67.5% 又离开了。",
    ],
  },
  {
    id: "ctrl/cem", from: "representative-atlas",
    t: "CEM 与滚动规划", en: "Cross-Entropy Method & MPC",
    remember: [
      "CEM：从高斯采样 → 打分 → 取精英 → 用精英重新拟合均值和方差 → 迭代。这里是 300 个样本、10 轮、30 个精英。",
      "MPC：只执行第一个动作，拿到新观测后重新规划。",
      "单步贪心能用的前提：代价本身就是「离目标还有多远」。",
    ],
  },
  {
    id: "ctrl/belief-planning", from: "representative-atlas",
    t: "多假设下的规划", en: "Planning over a Belief",
    remember: [
      "状态不确定时，最小化期望代价 $\\mathbb{E}_{u\\sim b}[c(u,a)]$，不要对均值或最可能的状态做规划。",
      "两个等权假设在 $\\mu\\pm d$ 时，任何 $|\\Delta|\\le d$ 的移动，期望距离都是 $d$，所以期望代价不会误以为「已经到了」。",
      "动作 → 坐标位移的映射 $\\Delta u = A\\,\\Delta x$ 必须对：打乱它是唯一让 Dwell30 显著下降的消融。",
    ],
  },
  {
    id: "method/closed-loop-eval", from: "representative-atlas",
    t: "闭环评测怎么设计", en: "Designing a Closed-Loop Evaluation",
    remember: [
      "到达、停留、终点是三种能力，分开报告。固定步数又没有 stop 动作时，终点指标才说明「停没停住」。",
      "加参考策略给任务定界：oracle（能不能到）、random、no motion（起点分布本身「白送」多少）。",
      "所有策略共享初始位姿和成像参数，才能做配对比较。",
      "消融一次只断一个对应关系，看它坏在「到不了」还是「停不住」。",
    ],
  },
  {
    id: "method/statistics", from: "representative-atlas",
    t: "配对、Bootstrap 与 Holm", en: "Paired Tests, Bootstrap & Holm",
    remember: [
      "按病人划分数据。所有选择都在验证集上做，跑测试之前先写下主比较。",
      "病人是聚类单位：先在每个病人内部汇总，再按病人做 bootstrap，求配对差的置信区间。",
      "Holm：p 值从小到大排，第 $i$ 个和 $\\alpha/(m-i+1)$ 比，第一次不过就停。",
      "95% CI 不含 0（未校正），但 Holm 校正后 p = 0.068：照实写「校正后不显著」。",
    ],
  },

  // ---- SwarmAgentic ----
  {
    id: "math/pso", from: "swarmagentic",
    t: "粒子群优化", en: "Particle Swarm Optimization",
    remember: [
      "速度 = 惯性 + 向自己的历史最优靠 + 向群体最优靠；新位置 = 旧位置 + 速度。",
      "PSO 只需要两个原语：比较两个解（diff）和按差去改（apply）。能实现这两个，就能跑 PSO。",
      "种群方法每轮要评估 N 次；一次评估要几个小时时，贝叶斯优化通常更划算。",
    ],
  },
  {
    id: "llm/language-space-optimization", from: "swarmagentic",
    t: "语言空间里的优化", en: "Optimization in Language Space",
    remember: [
      "速度是一份自然语言的修改计划；$p^* - x$ 和 $g - x$ 变成以当前缺陷为锚的 LLM 比较。",
      "先计划、再修改，分两次调用：每一步都是人能读懂的「缺陷 → 修改」清单。",
      "诊断错误比解决任务容易，所以更弱的模型也能当优化器。",
      "论文里的 PSO 系数和随机因子，代码里都没有：多样性几乎全来自初始化温度。",
    ],
  },
  {
    id: "agent/system-as-particle", from: "swarmagentic",
    t: "整个团队是一个粒子", en: "A Whole Team as One Particle",
    remember: [
      "一个粒子 = 一整套团队设计（角色 + 工作流 + 编译出的 forward 代码），不是单个智能体。",
      "角色 = (名字, 职责, 策略)，像职位名、岗位描述、操作手册。",
      "粒子之间是竞争加抄作业，智能体之间才是分工协作。",
      "小修改（只改策略）是利用，增删角色和步骤是探索；新增角色门槛要高，防止团队臃肿。",
    ],
  },
  {
    id: "agent/failure-driven-update", from: "swarmagentic",
    t: "从失败中学习", en: "Failure-Driven Updates",
    remember: [
      "先诊断再更新：失败案例 → 7 类有名字的缺陷 → 每条修改对准一个缺陷。",
      "惯性项换成失败项：针对某缺陷的修改，下一轮缺陷还在，就算失败，不许再提。",
      "论文主打的失败项，消融只掉 0.4；去掉协作结构重组掉 2.1。叙事重要 ≠ 实证重要。",
    ],
  },
  {
    id: "agent/automated-agent-design", from: "swarmagentic",
    t: "自动设计智能体", en: "Automated Agentic-System Design",
    remember: [
      "完全自动 = 从零生成 + 自我优化功能 + 自我优化协作；ADAS 只差第一条（七个手写种子）。",
      "「完全自动」仍需要可计算的目标函数和人写的元框架（格式、缺陷类型、编辑算子）。",
      "ADAS 把所有历史设计塞进 prompt，上下文会被撑满，搜索也偏向新奇而不是更好。",
    ],
  },
  {
    id: "method/search-eval", from: "swarmagentic",
    t: "搜索式设计怎么评测", en: "Evaluating Search-Based Design",
    remember: [
      "用稠密的代理指标搜（微平均通过率），用严格的指标报告（最终通过率）。",
      "相对提升要看底数：+261.8% 是 32.2 vs 8.9，绝对值 23.3 个点。",
      "消融要看噪声底：同一个设置跑出 6.4 和 6.7，差 0.3，和被测的效应差不多大。",
      "放一个方法不该有太大优势的控制任务，帮读者看清适用范围。",
    ],
  },
  {
    id: "infra/llm-loop-engineering", from: "swarmagentic",
    t: "LLM 回路的工程", en: "Engineering an LLM-in-the-Loop Search",
    remember: [
      "LLM 在循环里的搜索，大头是适应度评估：每轮每个粒子都要把整个系统重跑一遍。",
      "任何 `exec()` LLM 生成代码的系统都要进沙箱。",
      "长搜索要能断点续跑；所有「重试直到对上」的循环都要设上限。",
    ],
  },

  // ---- SurgAgentGym ----
  {
    id: "med/surgical-scene", from: "surgagentgym",
    t: "外科场景理解", en: "Surgical Scene Understanding",
    remember: [
      "三层任务：空间（器械、定位、三元组）、时间（阶段、动作）、工作流（CVS、技能评估）。",
      "CVS 三条标准要一起判断、还要持续成立，是逐帧感知最难的任务。",
      "阶段和三元组里的动作是跨帧的事实，单看一帧只能猜。",
      "医学专用 MLLM 在需要专家判断的任务上没有比通用开源模型更好。",
    ],
  },
  {
    id: "agent/harness", from: "surgagentgym",
    t: "Agent 与 harness", en: "From LLM to Agent: the Harness",
    remember: [
      "agent = LLM + 状态 + 工具接口 + 解析约束 + 交互循环；工具调用只是看得见的那部分。",
      "harness 的四个维度：工具、控制流、状态、校验。能用 API 不用 LLM，能用单次 LLM 不用子智能体。",
      "预算加强制交卷，才把开放的工具循环变成能打分的 episode。",
      "prompt 叫它用工具，工具调用频率测的就是听不听话，不是需不需要。",
    ],
  },
  {
    id: "agent/hybrid-retrieval", from: "surgagentgym",
    t: "混合检索：KB 与 RAG", en: "Bounded Hybrid Retrieval",
    remember: [
      "KB 是底座（存、索引、搜），RAG 是用法（怎么喂给模型）。",
      "BM25 和向量检索用 RRF 按排名融合：$\\sum 1/(k + \\text{rank})$，不要直接加原始分数。",
      "纯文本重排序器看不见图：多模态任务里，高相关可能是自信的错误证据。",
      "余弦去重挡得住改写，挡不住换个说法重复同一个意图。",
    ],
  },
  {
    id: "agent/failure-attribution", from: "surgagentgym",
    t: "轨迹失败归因", en: "Trajectory-Level Failure Attribution",
    remember: [
      "失败的智能体和成功的一样频繁地调用工具（2.36 vs 2.32 次）：差距在怎么用证据。",
      "怪策略之前，先看动作空间能不能表达需要的观测。",
      "工具效应要按模型报告：平均会让方向相反的行为互相抵消。",
      "按成功 / 失败分组的统计是相关，不是因果。",
    ],
  },
  {
    id: "llm/structured-output", from: "surgagentgym",
    t: "结构化输出", en: "Structured-Output Contracts",
    remember: [
      "把怎么调用模型（载体）和怎么约束、解析输出（协议）分开。",
      "解码层的约束（vLLM guided_json）让任何开源模型都能稳定输出动作，代价是压制它原生的思考。",
      "外层动作结构严格约束，每个工具的参数单独校验。",
      "guided JSON 适合约束最终结果，不适合替代原生的 tool-calling 协议。",
    ],
  },
  {
    id: "llm/mllm-basics", from: "surgagentgym",
    t: "MLLM 怎么看图", en: "How Multimodal LLMs See",
    remember: [
      "MLLM = 视觉编码器 → 投影层 → LLM；视觉 token 和文本 token 在自注意力里融合。",
      "图越多、分辨率越高，视觉 token 越多，上下文主要被图像吃掉。",
      "CLIP 式双编码器没有解码器，能检索、不能当智能体。",
      "Set-of-Mark：分割模型管「在哪」，MLLM 管「是什么」。",
    ],
  },
  {
    id: "method/metrics-shortcuts", from: "surgagentgym",
    t: "指标的捷径", en: "Metrics that Resist Shortcuts",
    remember: [
      "准确率高、κ ≈ 0，说明模型学到的是标签分布，不是任务。",
      "LLM 没有校准过的置信度，检测类任务用 F1@IoU，不用 mAP。",
      "先在数据集之间取宏平均，否则最大的那个数据集就成了整个 benchmark。",
      "解析失败记为空预测、留在分数里，不要悄悄丢掉。",
    ],
  },
  {
    id: "method/hard-set-controls", from: "surgagentgym",
    t: "难例集、对照与统计", en: "Hard Sets, Controls & Clustered Statistics",
    remember: [
      "按模型失败挑出的难例集，必须有匹配对照，否则提升可能只是回归均值。",
      "按视频做 bootstrap，不按帧：同一个视频里的帧不独立。",
      "LLM 裁判和每个专家的一致性，不低于专家之间的一致性，才算可用。",
      "BH-FDR 控制显著结果里错的比例，比控制「至少错一个」的 Holm 宽松。",
    ],
  },
  {
    id: "infra/vllm-slurm", from: "surgagentgym",
    t: "在 SLURM 上跑 vLLM", en: "Serving MLLMs with vLLM on SLURM",
    remember: [
      "serve once, run many：模型服务一个作业，评测客户端另外的作业，客户端不要申请 GPU。",
      "vLLM 按剩余显存的比例预留 KV cache：给 CUDA graph 留余量，或者用 eager 模式。",
      "多模态请求的长度主要是图像 token：限制每次的帧数。",
    ],
  },

  // ---- PrivacyEvalClaw ----
  {
    id: "llm/privacy-leakage", from: "privacyevalclaw",
    t: "什么算隐私泄露", en: "What Counts as Privacy Leakage",
    remember: [
      "只要出现和某个人绑定的、具体的 PII 值就算泄露，不管是真的还是编的；只有 `[REDACTED]` 这类非值才不算。",
      "**不安全 ≠ 泄露**：教人入侵是安全问题，但没说出任何具体信息就不是隐私泄露。",
      "把「泄露」「帮助获取」「拒绝」拆成不同标签，很多分歧其实是大家在回答不同的问题。",
    ],
  },
  {
    id: "llm/jailbreak-guards", from: "privacyevalclaw",
    t: "越狱与守卫模型", en: "Jailbreaks & Guard Models",
    remember: [
      "越狱套路：拒绝抑制、角色扮演、场景嵌套、编码混淆、多语言。",
      "guard 分 prompt 级（判意图）和 response 级（判回答）；不同 guard 判「安全」的比例能差将近 3 倍。",
      "模型类评估器会「脑补」Base64 的解码结果，确定性解码器不会。",
    ],
  },
  {
    id: "agent/deliberative-agent", from: "privacyevalclaw",
    t: "审议式工具智能体", en: "Tool-Augmented Deliberative Agent",
    remember: [
      "当法官，不当分类器的聚合器：保持一个假设，下一步调用最能回答关键疑点的工具。",
      "证据优先级：回答原文 > 定向推理 > PII 检测 > guard；越确定性、越接近原文越可信。",
      "写清停止规则：结论稳定就停，不为了完整而多调工具。",
      "审议很贵：每条约 114K token，直接判只要约 2.3K。",
    ],
  },
  {
    id: "agent/disagreement", from: "privacyevalclaw",
    t: "工具打架时怎么办", en: "Reconciling Conflicting Tools",
    remember: [
      "错误相关、定义不同的评估器，投票救不了：12 个 guard 投票和人的 κ = −0.022。",
      "分歧是信号：问「回答原文到底支持哪个说法？请引用」，而不是再问一遍「安全吗」。",
      "确定性工具（解码器、正则）是最客观的裁判。",
    ],
  },
  {
    id: "method/agreement-kappa", from: "privacyevalclaw",
    t: "一致性与 Cohen's κ", en: "Agreement & Cohen's Kappa",
    remember: [
      "$\\kappa = (p_o - p_e)/(1 - p_e)$：扣掉碰巧一致的部分；0 = 随机，1 = 完全一致。",
      "标签不平衡时一致率会骗人：永远答「泄露」也有 62% 一致率，但 κ = 0。",
      "评估器之间一致 ≠ 评估器是对的；只有和人工标签比才是在量对不对。",
    ],
  },
  {
    id: "method/pairwise-judge", from: "privacyevalclaw",
    t: "成对评审：过程 vs 结论", en: "Pairwise LLM-as-Judge",
    remember: [
      "成对评审适合比较整段推理轨迹，按胜 / 负 / 平报告，平局不算半个胜。",
      "裁判会奖励看得见的调查：过程胜率 > 98%，结论胜率只有约一半，所以还要对人工标签。",
      "LLM 裁判的三个偏差：位置、长度、自我偏好。",
    ],
  },
  {
    id: "infra/vllm-inference", from: "privacyevalclaw",
    t: "vLLM 批量推理", en: "Batch Inference with vLLM",
    remember: [
      "vLLM 快在 PagedAttention（分页管理 KV cache）和 continuous batching。",
      "OOM 多半是 KV cache 预留太多：`max_model_len` 按任务需要设，别按模型上限。",
      "截断 judge 输入时，保留问题的结尾和回答的开头。",
    ],
  },
  {
    id: "infra/parallelism-sft", from: "privacyevalclaw",
    t: "推理并行与 LoRA 微调", en: "Parallelism & LoRA SFT",
    remember: [
      "vLLM 的 TP/PP/DP 和训练的 DDP/FSDP 是两套系统，只通过数据和 LoRA 权重连在一起。",
      "`device_map=\"auto\"` 不是分布式训练，多卡训练用 `accelerate launch` / `torchrun`。",
      "SFT 只在回答上算 loss：prompt 部分的 label 设成 −100。",
      "LoRA：$W' = W + \\frac{\\alpha}{r}BA$，$B$ 初始化为 0。",
    ],
  },
];
