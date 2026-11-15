> [!one] 世界模型（world model）就是机器人脑中的"模拟器"：学会 $p(s_{t+1} \mid s_t, a_t)$ 之后，机器人就能在想象里试错。它可以用来规划（MPC）、在"梦"里训练策略（Dreamer），也可以用来生成数据、评估策略，而不必每次都拿真机去撞。

## 直觉

人伸手推桌边的杯子之前，脑子里已经"看到"它掉下去摔碎了，所以不会真的去推。这种**在行动之前预测后果**的能力，就是世界模型想给机器人的东西。

机器人为什么特别需要它？因为真实交互又贵、又慢、又危险。Model-free RL（如 PPO、SAC）动辄要几百万步交互，在仿真里没问题，在真机上就是几周的磨损和无数次摔倒。有了一个可以"想象"的模型，就能把大部分试错搬到脑子里（详见 [Model-based RL](#/learn/rl/mbrl)）。世界模型对机器人主要有三种用法：

1. **在想象中规划（planning / MPC）**：每一步都在模型里试几百条动作序列，挑最好的执行；
2. **在想象中学策略**：用模型生成大量虚拟轨迹来训练 actor-critic（Dreamer），真机只用来采数据、校准模型；
3. **数据生成、仿真与策略评估**：像"可学习的仿真器"一样生成训练数据，或者给策略打分，代替昂贵的真机 rollout。

注意"世界模型"这个词现在被用得很宽：RL 里的 latent dynamics model、能生成可交互视频的大模型、LeCun 提倡的 JEPA，都叫世界模型。这一课的任务就是把它们放在同一张地图上。

## 形式化

**基本定义**：给定状态和动作，预测下一状态的分布：

$$
s_{t+1} \sim p_\phi(s_{t+1} \mid s_t, a_t)
$$

机器人拿到的不是真实状态，而是高维观测 $o_t$（图像、本体感知），所以实际上通常在**隐空间（latent space）**里建模：编码器 $z_t = \mathrm{enc}(o_{\le t})$，隐动力学 $z_{t+1} \sim p_\phi(z_{t+1} \mid z_t, a_t)$，再加上可选的解码器 $p_\phi(o_t \mid z_t)$ 和奖励头 $\hat{r}(z_t, a_t)$。

### RSSM（Dreamer 系列）

Recurrent State-Space Model 把状态拆成确定性部分 $h_t$（GRU 隐状态，负责长程记忆）和随机部分 $z_t$（负责不确定性）。DreamerV3 的写法（$x_t$ 是观测）：

| 组件 | 公式 | 作用 |
| --- | --- | --- |
| Sequence model | $h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1})$ | 确定性的循环状态 |
| Encoder（posterior） | $z_t \sim q_\phi(z_t \mid h_t, x_t)$ | 看到观测后的随机状态 |
| Dynamics（prior） | $\hat{z}_t \sim p_\phi(\hat{z}_t \mid h_t)$ | 不看观测，预测随机状态 |
| Reward / continue | $\hat{r}_t \sim p_\phi(\hat{r}_t \mid h_t, z_t)$，$\hat{c}_t \sim p_\phi(\hat{c}_t \mid h_t, z_t)$ | 预测奖励和 episode 是否继续 |
| Decoder | $\hat{x}_t \sim p_\phi(\hat{x}_t \mid h_t, z_t)$ | 重建观测，提供稠密的学习信号 |

训练时 posterior 能看到真实观测 $x_t$，prior 看不到。想象（imagination）时只能用 prior 往前滚。损失函数：

$$
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}\Big[\sum_t \beta_{\text{pred}}\mathcal{L}_{\text{pred}} + \beta_{\text{dyn}}\mathcal{L}_{\text{dyn}} + \beta_{\text{rep}}\mathcal{L}_{\text{rep}}\Big]
$$

$$
\mathcal{L}_{\text{dyn}} = \max\big(1,\ \mathrm{KL}[\mathrm{sg}(q_\phi(z_t \mid h_t, x_t))\ \|\ p_\phi(z_t \mid h_t)]\big),\qquad \mathcal{L}_{\text{rep}} = \max\big(1,\ \mathrm{KL}[q_\phi(z_t \mid h_t, x_t)\ \|\ \mathrm{sg}(p_\phi(z_t \mid h_t))]\big)
$$

$\mathcal{L}_{\text{pred}}$ 是重建观测、奖励和 continue flag 的负对数似然；$\mathrm{sg}$ 是 stop-gradient；$\max(1, \cdot)$ 是 free bits（KL 小于 1 nat 时不再压缩）。DreamerV3 取 $\beta_{\text{pred}} = 1,\ \beta_{\text{dyn}} = 0.5,\ \beta_{\text{rep}} = 0.1$。直觉：**主要让 prior 去追 posterior**（学会预测），只轻微地让 posterior 向 prior 靠（表征别太难预测）。

### 用模型规划：MPC

在模型里优化未来 $H$ 步动作，加一个终端价值来覆盖 $H$ 之后的回报：

$$
a^*_{t:t+H-1} = \arg\max_{a_{t:t+H-1}}\ \mathbb{E}\Big[\sum_{k=0}^{H-1}\gamma^k\,\hat{r}(z_{t+k}, a_{t+k}) + \gamma^H\,\hat{V}(z_{t+H})\Big],\qquad z_{t+k+1} \sim p_\phi(\cdot \mid z_{t+k}, a_{t+k})
$$

只执行 $a^*_t$，下一步再重新规划（receding horizon）。神经网络模型不好求解析梯度，常用的是**采样式优化器**：

```python
# CEM / MPPI 的骨架
mu, std = zeros(H, d_a), ones(H, d_a)
for it in range(n_iters):
    A = mu + std * randn(N, H, d_a)            # 采 N 条动作序列
    R = rollout_return(model, z_t, A)          # 在 latent 里滚 H 步，算回报（加终端 V）
    # CEM：取 top-K elites，直接重新拟合均值和方差
    # MPPI：按回报做 softmax 加权，w_i = exp((R_i - R.max()) / lam) / sum(...)
    w = softmax(R / lam)                       # MPPI 版本
    mu  = (w[:, None, None] * A).sum(0)
    std = sqrt((w[:, None, None] * (A - mu) ** 2).sum(0))
execute(mu[0]); mu = shift(mu)                 # 只执行第一步，warm-start 下一次规划
```

### 复合误差

假设模型每步误差不超过 $\epsilon$，真实动力学是 $L$-Lipschitz 的，那么 $e_{k+1} \le L e_k + \epsilon$，展开得到：

$$
e_H \le \epsilon\,\frac{L^H - 1}{L - 1}
$$

$L > 1$（不稳定系统、接触丰富的场景）时误差随 horizon **指数增长**。这是所有世界模型的核心约束：**想象得越远，越不可信。** 应对办法包括短 horizon 加 value bootstrap（Dreamer、TD-MPC）、每步重新规划、多步预测 loss、从真实状态出发做短分支 rollout（MBPO），以及用 ensemble 估计不确定性并惩罚不确定的区域。

## 方法：四个家族

### 1. 重建式 latent dynamics：World Models → Dreamer

（详见 [Latent 世界模型](#/learn/wm/latent) 和[在想象中学策略](#/learn/wm/policy)。）

- **World Models（Ha & Schmidhuber, 2018）**：V（VAE 把图像压成 $z$）+ M（MDN-RNN 预测下一个 $z$ 的混合高斯分布）+ C（一个很小的线性控制器，用 CMA-ES 优化）。在 VizDoom 里控制器**完全在梦里训练**再迁移回真实环境。为了不让控制器钻模型的空子，作者调高了采样温度，让梦变得更"难"。
- **PlaNet（2019）**：提出 RSSM，用 CEM 在 latent 里规划，不训练策略网络。
- **Dreamer V1–V3**：在 latent 里想象约 15 步，训练 actor-critic，价值用 $\lambda$-return。V1 把价值的解析梯度穿过模型反传；V2 用离散 latent（32 个 32 类的 categorical）；**V3 一套固定超参跑通 150 多个任务**，并且是第一个在 Minecraft 里不靠人类数据、从零挖到钻石的算法。
- **DayDreamer（2022）**：直接在真机上跑 Dreamer，不用仿真器。A1 四足机器人从零学会翻身、站起和行走大约只用了 1 小时；机械臂从图像和稀疏奖励学会 pick-and-place。所有实验用同一套超参。

### 2. 任务导向的隐式模型：TD-MPC / TD-MPC2

不重建像素，模型只为"做决策"服务（详见 [TD-MPC](#/learn/wm/tdmpc)）。TD-MPC2 的组件：编码器 $z = h(s)$、隐动力学 $z' = d(z, a)$、奖励 $\hat{r} = R(z, a)$、终端价值 $\hat{q} = Q(z, a)$、策略先验 $\hat{a} = p(z)$。训练目标是三项（沿时间按 $\lambda^t$ 衰减加权）：

$$
\mathcal{L} = \mathbb{E}\Big[\sum_{t}\lambda^t\Big(\big\|z'_{t+1} - \mathrm{sg}(h(s_{t+1}))\big\|_2^2 + \mathrm{CE}(\hat{r}_t, r_t) + \mathrm{CE}(\hat{q}_t, q_t)\Big)\Big],\qquad q_t = r_t + \gamma\,\bar{Q}(z'_{t+1}, p(z'_{t+1}))
$$

$\bar{Q}$ 是 $Q$ 的 EMA。latent 用 SimNorm（分组 softmax）约束，避免数值发散。规划用 MPPI，horizon 默认只有 3，后面靠 $Q$ bootstrap，策略先验额外贡献一部分候选动作序列来加速收敛。TD-MPC2 在 104 个连续控制任务上评测，最大的多任务 agent 有 317M 参数。

### 3. 视频生成世界模型

（详见[视频世界模型](#/learn/wm/video)。）

- **UniSim（2023）**：用动作条件的视频扩散模型模拟真实世界交互，既接受"打开抽屉"这种高层文本动作，也接受低层控制。在里面训练的 VLM 规划器和 RL 策略能零样本迁移到真机。
- **Genie（2024）**：11B 参数，在无标注的互联网游戏视频上训练。核心是 **latent action model**：无监督地从相邻帧之间推断出一个很小的离散动作码本，于是没有动作标签的视频也能变成"可交互"的环境。组成部分是时空视频 tokenizer、latent action model 和自回归动力学模型。后续的 Genie 版本扩展到了 3D 和实时交互。
- **NVIDIA Cosmos（2025）**：world foundation model 平台，包含视频数据管线、tokenizer、扩散和自回归两类预训练世界模型，开放权重，可以后训练成面向机器人或自动驾驶的、带动作条件的模型。
- **另外两种用法**：UniPi 先生成"任务完成的视频"，再用逆动力学模型（IDM）从视频里反推动作；GR00T N1 用视频模型生成的 neural trajectories 扩充训练数据。

### 4. JEPA：只在表征空间里预测

LeCun 的立场论文（*A Path Towards Autonomous Machine Intelligence*, 2022）认为：世界上大量细节本质上不可预测（树叶怎么晃、水波纹），逐像素生成是在浪费容量。应该**在表征空间里预测**：用上下文 $x$ 的表征 $s_x$ 去预测目标 $y$ 的表征 $\bar{s}_y$，即 $\min D\big(\mathrm{Pred}(s_x),\ \mathrm{sg}(\bar{s}_y)\big)$。预测器还可以额外以动作或隐变量为条件，目标编码器通常是在线编码器的 EMA 版本（详见 [JEPA](#/learn/wm/jepa)）。

**V-JEPA 2（2025）**：先在超过 100 万小时的互联网视频上自监督预训练编码器；再冻结编码器，在约 62 小时的 DROID 机器人视频（只有末端状态和动作，没有任务或奖励标注）上训练一个约 300M 参数、以动作为条件的预测器，得到 **V-JEPA 2-AC**。规划时给一张目标图像，用 CEM 最小化能量：

$$
\mathcal{E}(\hat{a}_{1:T}) = \big\| P(\hat{a}_{1:T};\ s_k, z_k) - z_g \big\|_1
$$

其中 $z_k$ 是当前帧的表征，$s_k$ 是当前末端状态，$z_g$ 是目标图像的表征，$P$ 是动作条件预测器滚出的 $T$ 步之后的表征。执行第一个动作后重新规划。它被零样本部署到两个实验室的 Franka 机械臂上做抓取和 pick-and-place，这两个实验室都不在 DROID 里。论文报告每个动作的规划约 16 秒，而用 Cosmos 做同样的规划约 4 分钟。

### 对比

| 家族 | 代表 | 在哪里预测 | 训练信号 | 典型用途 | 速度 | 主要风险 |
| --- | --- | --- | --- | --- | --- | --- |
| 重建式 latent | World Models、PlaNet、Dreamer、DayDreamer | 紧凑 latent，可解码回图像 | 重建 + 奖励 + KL | 想象中学策略、规划 | 快，可并行想象大量轨迹 | 重建把容量浪费在无关细节上 |
| 任务导向隐式 | TD-MPC、TD-MPC2 | latent，不解码 | latent consistency + 奖励 + TD | 在线 MPC | 很快 | 和奖励绑定；没有奖励和价值信号撑着，latent 容易退化 |
| 视频生成 | UniSim、Genie、Cosmos | 像素 / 视频 token | 扩散或自回归生成 | 数据生成、策略评估、交互式仿真 | 慢，秒到分钟级 | 物理不一致，动作条件弱，幻觉 |
| JEPA | V-JEPA 2-AC | 自监督表征空间 | 表征预测（EMA target） | 目标图像条件的零样本规划 | 中等 | 表征塌缩；目标要用图像表达；长程困难 |

### 像素还是 latent？

这是整个领域最核心的取舍：

- **重建像素**：监督信号稠密，表征不会塌缩，也方便人看模型"想"了什么。但模型要花大量容量去画背景纹理和光照，而和任务相关的关键信息（夹爪和物体之间那几毫米）在像素 loss 里占比很小。
- **只在 latent 里预测**：容量集中在可预测、与决策相关的部分。但如果只要求"预测下一个 latent"，最省事的解就是让编码器把一切都映射成常数，这样预测误差恒为 0，也就是**表征塌缩（collapse）**。所以需要额外的锚：TD-MPC 用奖励和价值，JEPA 用 EMA target encoder、stop-gradient 或者方差/协方差正则。

### 视频世界模型的两个硬伤

1. **动作条件弱**：互联网视频没有动作标签，只能用文本或 latent action 来条件化。文本太粗，latent action 不一定对应机器人的真实控制量。要做精确的操作规划，必须有带动作标签的机器人交互数据。
2. **物理一致性**：视频看起来对，不代表物理对。常见问题有物体凭空出现或消失、穿模、手碰到之前物体就动了、质量不守恒。评价视频生成的 FVD 等指标几乎不衡量这些。用它来评估策略时，如果模型在接触时刻"脑补"出成功，评估就会系统性地偏乐观。

### 和物理仿真器、数字孪生的关系

（仿真器见[仿真引擎](#/learn/sim/engines)，两者的结合见[物理与世界模型](#/learn/wm/physics)。）

| | 解析物理仿真器（MuJoCo、Isaac 等） | 学习的世界模型 |
| --- | --- | --- |
| 知识来源 | 人写的物理方程 + 人工建模的资产 | 从数据中学 |
| 准确性 | 刚体很准；接触、摩擦、软体、流体只能近似 | 数据分布内可以很真实，包括难以解析建模的现象；分布外不可靠 |
| 视觉 | 渲染有 sim-to-real gap | 视频模型可以做到照片级 |
| 搭建成本 | 每个新场景都要建模、调参 | 有数据就能学，但需要大量数据和算力 |
| 速度 | GPU 上可以并行成千上万个环境 | latent 模型快，视频模型慢 |
| 可控性 | 状态完全可读写，能复位、能注入扰动 | 状态是隐变量，很难精确设定初始条件 |

两者不是二选一：可以在解析模型上学残差动力学，可以用真实场景重建出数字孪生（digital twin），也可以用仿真数据给世界模型做预训练。

### 开放问题

- **长程一致性**：几秒钟之后物体身份、遮挡后的物体恒存（object permanence）、场景布局往往就开始漂移。
- **精细接触**：抓取、插入这类毫米级、力主导的交互，单靠视觉很难预测，力觉和触觉怎么进入世界模型还没有定论。
- **动作接口**：互联网视频没有动作，机器人数据的动作空间又各不相同。latent action 能不能成为通用的"动作语言"，还是开放问题。
- **评测**：怎样衡量世界模型"物理上对不对、对决策有没有用"，目前还没有被广泛接受的标准。
- **速度与规模的矛盾**：越大越真实的视频模型越慢，在线规划需要的是又快又准的模型。蒸馏、在 latent 里规划、只在关键时刻调用大模型，都是正在探索的方向。

## 机器人视角

> [!robot] **按延迟预算选模型**：在线 MPC 每个控制周期都要做几百次 rollout，只有轻量的 latent 模型（TD-MPC2、Dreamer 一类）跑得动。视频世界模型目前主要用在**离线**场景：生成数据、评估策略、做交互式仿真。V-JEPA 2-AC 每个动作约 16 秒的规划时间，对静态的抓取放置还能接受，对动态任务就远远不够。

> [!robot] **奖励是真机上最难的部分**：Dreamer 和 TD-MPC 都需要奖励信号，而真机上没有现成的 reward function，只能自己做检测器、写规则或者人工标注。用目标图像条件的能量（V-JEPA 2-AC）或者示范数据来定义目标，是绕开奖励设计的两条路。

> [!robot] **数据要覆盖策略会去的地方**：世界模型只在训练数据的分布里可信。策略一旦开始利用模型，就会把状态推到模型没见过的区域。所以实际系统几乎都是 Dyna 式的闭环：用当前策略在真机上采数据 → 更新模型 → 在模型里改进策略 → 再采数据。DayDreamer 能在约 1 小时内学会走路，靠的就是这个在线循环。

> [!robot] **用世界模型评估策略之前先做校准**：拿几个已知真机成功率的策略，看世界模型给出的排序是否和真机一致。排序一致，才能用它来筛 checkpoint、做消融；绝对成功率数字一般不要直接信。

## 常见坑

> [!pitfall] **模型被利用（model exploitation）**：优化器或策略会专门去找模型预测错误、而且错得"有利"的动作，比如穿墙、物体自己飞到目标位置。规划回报很高，真机上一塌糊涂。解决办法：用 ensemble 估计不确定性并惩罚，缩短 horizon，限制动作范围（V-JEPA 2-AC 就把每步动作限制在一个小的 L1 球内），并持续用真实数据纠正模型。

> [!pitfall] **想象得太远**：为了"看得更远"把 rollout horizon 拉到几十上百步，复合误差会让后半段完全失真。正确做法是短 horizon 加终端价值，每步重新规划。

> [!pitfall] **视频好看 ≠ 模型有用**：重建或生成质量（PSNR、FVD）高，不代表动作条件下的预测准确，更不代表规划有效。要直接评估下游指标：规划成功率，以及策略评估结果和真机结果的相关性。

> [!pitfall] **无重建模型的表征塌缩**：自己实现 latent-only 的模型时，如果 target 没有 stop-gradient、没有 EMA，也没有奖励或价值这样的锚，loss 会很快降到接近 0，但 latent 什么信息都没有。监控 latent 各维度的方差，是发现塌缩最简单的办法。

## 自测

> [!quiz] Dreamer 为什么只想象约 15 步，再接一个 value function，而不是直接想象到 episode 结束？
> 因为复合误差：每步的预测误差会累积，$L > 1$ 时还会指数放大，几十步之后的想象轨迹已经不可信，用它训练策略只会学到模型的 bug。短 horizon 内模型还算准，horizon 之外的回报交给 critic 用 $\lambda$-return bootstrap 估计。critic 是在大量想象数据上训练的，误差不会沿时间累积。这和 TD-MPC 用 horizon 3 加终端 $Q$ 是同一个思路。

> [!quiz] TD-MPC2 不重建像素也不会塌缩，为什么？JEPA 又靠什么防止塌缩？
> TD-MPC2 的 latent 必须能预测奖励，还要支撑 TD 价值学习。如果把所有状态都映射成同一个点，就无法区分高回报和低回报的状态，奖励和价值 loss 会很大，所以任务信号就是锚；latent consistency 的 target 还加了 stop-gradient。JEPA 没有奖励，纯自监督，所以要靠结构性手段：EMA target encoder 加 stop-gradient（非对称结构），或者 VICReg 一类的方差/协方差正则，强迫表征保持信息量。

> [!quiz] 你想用一个视频世界模型代替真机来评估 10 个抓取策略。可能出什么问题？怎么验证它可信？
> 问题有：（1）接触物理不准，模型可能"脑补"抓取成功，系统性高估；（2）动作条件弱，不同策略的细微动作差异在视频里体现不出来，区分度低；（3）策略本身如果在世界模型上调过，会过拟合模型的偏差；（4）长程 rollout 里的复合误差。验证方法：选几个已知真机成功率、性能有高有低的策略，看世界模型给出的排序和真机排序的相关性；检查失败案例（滑落、碰撞）能不能被模型复现；只把它用于相对比较，不报告绝对成功率。

> [!quiz] CEM 和 MPPI 都是采样式规划器，区别在哪？
> 都是从高斯分布采 $N$ 条动作序列，在模型里 rollout 算回报，再更新分布。CEM 硬选 top-K elites，用它们重新拟合均值和方差，等于把其余样本的权重设为 0。MPPI 用回报的 softmax 做软加权，$w_i \propto \exp(R_i / \lambda)$，温度 $\lambda$ 控制贪心程度：$\lambda \to 0$ 时接近只取最优样本，$\lambda$ 大时接近平均。MPPI 利用了所有样本的信息，更新更平滑。实践中两者都会做 warm-start（把上一次的解平移一步作为初值），TD-MPC 系列用的是加入 elite 选择的 MPPI 变体。

> [!quiz] 要为一个刚体抓取任务训练策略，什么时候选物理仿真器，什么时候选学习的世界模型？
> 刚体、几何已知、需要海量并行交互（比如 RL 训练几亿步）时选物理仿真器：快、可控、能复位，接触模型对刚体也够用，用 domain randomization 应对 sim-to-real 差距。物体难以建模（布料、液体、可变形物）、需要照片级视觉、或者已经有大量真实交互数据时，学习的世界模型更有优势。实践中常常组合使用：用仿真预训练，用真实数据做修正；或者用世界模型生成仿真很难覆盖的视觉多样性。

## 延伸阅读

- [World Models 交互式论文](https://worldmodels.github.io) — 可以直接在网页里玩"梦中的"赛车和 Doom，建立直觉最快
- [World Models（arXiv:1803.10122）](https://arxiv.org/abs/1803.10122) — V-M-C 架构的原始论文
- [DreamerV3（arXiv:2301.04104）](https://arxiv.org/abs/2301.04104) — RSSM 和一套固定超参背后的稳定性技巧（symlog、free bits、KL 平衡）
- [DayDreamer（arXiv:2206.14176）](https://arxiv.org/abs/2206.14176) — 世界模型在四种真实机器人上的实测
- [TD-MPC（arXiv:2203.04955）](https://arxiv.org/abs/2203.04955) 与 [TD-MPC2（arXiv:2310.16828）](https://arxiv.org/abs/2310.16828) — 不重建、为决策服务的隐式世界模型
- [UniSim（arXiv:2310.06114）](https://arxiv.org/abs/2310.06114) — 动作条件的视频模型当作真实世界模拟器
- [Genie（arXiv:2402.15391）](https://arxiv.org/abs/2402.15391) — 从无标注视频里学 latent action
- [Cosmos（arXiv:2501.03575）](https://arxiv.org/abs/2501.03575) — 面向 physical AI 的开放世界基础模型平台
- [V-JEPA 2（arXiv:2506.09985）](https://arxiv.org/abs/2506.09985) — 表征空间世界模型做零样本机器人规划

> [!mine] 我的笔记：学完后在这里写下自己的理解、疑问和实验记录。
