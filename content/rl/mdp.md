> [!one] MDP 把"机器人在世界里试错"抽象成一个循环：看状态 → 做动作 → 拿奖励 → 进入下一个状态；RL 要找的，是让"未来折扣奖励之和"的期望最大的那个策略。

## 直觉

想教一只四足机器人走路，你写不出"左前腿先抬 3 cm 再……"这样的规则，因为地形、负载、摩擦每次都不一样。RL 的思路是：**只规定"什么算好"，让机器人自己试出"怎么做"**。前提是先把问题写成所有 RL 算法都认的标准格式，这个格式就是马尔可夫决策过程（Markov Decision Process, MDP）。

MDP 描述的是一个 agent–environment loop（智能体–环境交互回路）：

1. agent（这里就是策略网络）在时刻 $t$ 看到状态 $s_t$；
2. 输出动作 $a_t$，比如 12 个关节的目标角度；
3. environment（仿真器或真实世界）推进一个控制周期，返回新状态 $s_{t+1}$ 和一个标量奖励 $r_t$；
4. 重复。

这个循环里有三件事要记住：

- **奖励是标量**。"走得快、走得稳、别抖、省电"这些愿望，最后都得压成一个数。reward 设计之所以难，根源就在这里（见 [奖励设计](#/learn/loco/reward)）。
- **奖励是延迟的**。这一步抬腿是好是坏，可能要过几步、看摔没摔才知道。所以目标是"未来奖励之和"，不是"这一步的奖励"。
- **环境是随机的**。同一个动作，换一种地面摩擦或电机温度，结果就不同。所以一切都要写成概率分布和期望。

**马尔可夫性**（Markov property）是整个框架的地基：只要"现在"描述得足够完整，未来就只取决于现在，与过去无关。对刚体系统，"位置 + 速度"才是完整状态（牛顿第二定律是二阶 ODE），只给位置不给速度就不是马尔可夫的。真机上"状态足够完整"几乎从不成立，后面讲 POMDP 时会细说。

## 形式化

一个 MDP 是元组 $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$，再加上初始状态分布 $\rho_0$：

| 符号 | 含义 | 四足例子 |
| --- | --- | --- |
| $\mathcal{S}$ | 状态空间 (state space) | 机身位姿与速度、关节角与角速度、地形…… |
| $\mathcal{A}$ | 动作空间 (action space) | $\mathbb{R}^{12}$，关节目标位置偏移 |
| $P(s' \mid s, a)$ | 转移概率 (transition dynamics) | 物理世界或仿真器，再加底层 PD 控制器 |
| $R(s, a)$ | 奖励函数 (reward function) | 速度跟踪减去各种惩罚 |
| $\gamma \in [0, 1)$ | 折扣因子 (discount factor) | 常用 0.99 |
| $\rho_0(s_0)$ | 初始状态分布 | 默认站姿附近随机初始化 |

马尔可夫性写成公式：

$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots, s_0) = P(s_{t+1} \mid s_t, a_t)
$$

### 策略

策略（policy）是从状态到动作的映射：

- 确定性策略（deterministic policy）：$a = \mu_\theta(s)$；
- 随机策略（stochastic policy）：$a \sim \pi_\theta(\cdot \mid s)$。

连续控制里最常用的是**对角高斯策略**（diagonal Gaussian policy）。网络输出均值 $\mu_\theta(s)$；标准差 $\sigma$ 通常是一个与状态无关、单独学习的参数向量：

$$
\pi_\theta(a \mid s) = \mathcal{N}\big(a ; \mu_\theta(s), \operatorname{diag}(\sigma^2)\big)
$$

$$
\log \pi_\theta(a \mid s) = -\sum_{i=1}^{d} \Big[ \frac{(a_i - \mu_i)^2}{2\sigma_i^2} + \log \sigma_i \Big] - \frac{d}{2}\log 2\pi
$$

训练时从分布里采样（为了探索），部署时一般直接用均值 $\mu_\theta(s)$。训练必须用随机策略有两个原因：一是要探索；二是策略梯度需要 $\nabla_\theta \log \pi_\theta$（见 [策略梯度](#/learn/rl/pg)）。顺带一提，在完全可观测的 MDP 中总存在确定性的最优策略，随机性只是为了学习；但在部分可观测时，无记忆的随机策略可能严格优于任何无记忆的确定性策略。

### 轨迹、回报与目标

轨迹（trajectory）$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$ 在策略 $\pi_\theta$ 下的概率是：

$$
p_\theta(\tau) = \rho_0(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)
$$

注意 $P$ 里没有 $\theta$。所以对 $\log p_\theta(\tau)$ 求梯度时动力学项直接消失，model-free RL 不需要知道物理模型。

回报（return）是从时刻 $t$ 开始的折扣奖励和，满足一个递推：

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k} = r_t + \gamma G_{t+1}
$$

RL 的目标是最大化期望回报：

$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\Big[\sum_{t=0}^{T} \gamma^t r_t\Big]
$$

所有 RL 算法都在用不同方式最大化 $J(\theta)$。[值函数方法](#/learn/rl/value) 先估计 $V^\pi(s) = \mathbb{E}[G_t \mid s_t = s]$ 和 $Q^\pi(s, a)$；[策略梯度](#/learn/rl/pg) 直接对 $J$ 求梯度；[PPO](#/learn/rl/ppo) 在策略梯度上加了稳定性约束。

### Episodic vs continuing

- **Episodic（分幕式）** 任务有自然终点，比如抓取成功或摔倒。
- **Continuing（持续式）** 任务没有终点，走路本质上就是持续任务。要让无穷和有限，需要 $\gamma < 1$。

训练时把 episode 截断在 20 s 左右，只是为了重置、换地形、换指令。这个"时间到"（time-out / truncation）**不是**真正的终止状态，处理方式和摔倒完全不同（见下文常见坑）。

### γ 的作用

因为 $\sum_k \gamma^k = 1/(1-\gamma)$，$\gamma$ 定义了一个有效视野（effective horizon）：

$$
H_{\text{eff}} \approx \frac{1}{1-\gamma} \text{ steps}
$$

$\gamma = 0.99$ 对应约 100 步，而且 $\gamma^{100} \approx 0.366 \approx e^{-1}$，也就是 100 步后的奖励权重衰减到大约 $1/e$。实践中 γ 与其说是"时间偏好"，不如说是**偏差–方差旋钮**：γ 越大越接近我们真正想要的长期目标，但回报的方差越大，值函数越难学。

### POMDP

真机上的问题几乎都是部分可观测马尔可夫决策过程（Partially Observable MDP, POMDP）。元组扩展为 $(\mathcal{S}, \mathcal{A}, P, R, \Omega, O, \gamma)$：agent 拿不到 $s_t$，只能看到观测 $o_t \sim O(\cdot \mid s_t)$。这时最优策略一般依赖整个历史 $h_t = (o_0, a_0, \dots, a_{t-1}, o_t)$，等价地也可以依赖信念（belief）$b_t(s) = p(s_t = s \mid h_t)$。

## 方法

### 把"四足走路"写成 MDP

下面以 legged_gym / Isaac Lab 风格的平地速度跟踪任务为例，数值都是典型配置，不是唯一答案：

| MDP 元素 | 具体内容 | 备注 |
| --- | --- | --- |
| 观测 $o_t$ | 机身角速度 (3)、投影重力 (3)、速度指令 $(v_x^{\mathrm{cmd}}, v_y^{\mathrm{cmd}}, \omega_z^{\mathrm{cmd}})$ (3)、关节位置减默认姿态 (12)、关节速度 (12)、上一步动作 (12) | 共 45 维。角速度和重力方向来自 IMU，关节量来自编码器 |
| 动作 $a_t$ | 12 维关节目标位置偏移：$q^{\mathrm{des}} = q_{\mathrm{default}} + k_a a_t$ | $k_a$ 是 action scale（如 0.25），由底层 PD 跟踪，见 [PD 控制](#/learn/ctrl/pd) |
| 转移 $P$ | 物理以 200 Hz 积分，每个子步算一次 PD 力矩；策略 50 Hz | 域随机化让 $P$ 变成一族分布，见 [域随机化](#/learn/loco/dr) |
| 奖励 $r_t$ | 跟踪项形如 $\exp(-\lVert v_{xy}^{\mathrm{cmd}} - v_{xy} \rVert^2 / \sigma)$，偏航角速度同理；正则项惩罚竖直速度、roll/pitch 角速度、力矩、关节加速度、动作变化率 $\lVert a_t - a_{t-1} \rVert^2$、非足端碰撞 | 详见 [奖励设计](#/learn/loco/reward) |
| 终止 | 躯干触地或倾角过大是真终止；超时（如 20 s）只是截断 | 两者对值函数的含义完全不同 |
| $\rho_0$ | 关节角在默认姿态附近随机，机身速度、地形、指令都随机 | 初始分布太窄，策略就只会"从站好开始走" |
| $\gamma$ | 0.99 | 50 Hz 下约 2 s 视野 |

表里几个设计背后的道理：

- **投影重力**（projected gravity）是把世界系重力方向 $(0, 0, -1)$ 转到机身坐标系：$g_b = R_{wb}^\top (0, 0, -1)^\top$。它编码了 roll 和 pitch，却没有欧拉角的奇异点，也不含偏航——机器人不需要知道自己朝东还是朝西。
- **为什么不给机身线速度？** legged_gym 原版观测里有 base linear velocity，但真机没有直接测它的传感器，只能用 IMU 积分加腿式里程计估计，噪声和漂移都大。很多 sim-to-real 工作直接拿掉它，或者用估计器、[teacher-student](#/learn/loco/teacher) 补回来。
- **为什么要给上一步动作？** 原因有两个。第一，奖励里有 $\lVert a_t - a_{t-1} \rVert^2$，如果 $a_{t-1}$ 不在观测里，奖励对 agent 来说就不是它所见状态的函数。第二，PD 目标本身是执行器的"内部状态"：关节角和角速度相同，上一拍目标不同，接下来的力矩就不同。
- **指令也是状态**。速度指令是外部给的目标，放进观测就成了目标条件策略（goal-conditioned policy），一个网络就能学会所有速度。

### 处理部分可观测

真机上看不到的量包括机身线速度、地形几何、摩擦系数、负载、电机温度与力矩常数、执行器延迟等。常见的处理办法：

1. **观测堆叠**（observation stacking）：把最近 $k$ 拍的 $(o, a)$ 拼起来作为输入。最简单，适合只需要短期记忆的场合，比如估计速度、感知打滑。
2. **循环网络**（RNN / LSTM / GRU）：用隐状态压缩无限长的历史，近似 belief。
3. **历史编码器**：用 MLP、1D-CNN 或 Transformer 编码较长的历史，回归出隐变量（例如 RMA 式的 adaptation module）。
4. **非对称 actor-critic**（asymmetric actor-critic）：critic 只在仿真里用，可以吃特权信息（真实速度、摩擦、地形高度）；actor 只用真机能拿到的观测。这样 critic 学得准，actor 部署也不受影响。

## 机器人视角

> [!robot] **γ 和控制频率是绑在一起的。** 有效视野按"步"算，换成真实时间要乘控制周期：$T_{\mathrm{eff}} = \Delta t / (1-\gamma)$。50 Hz、γ=0.99 时是 0.02 s × 100 = 2 s，大约 2–4 个步态周期，足够让策略"看到"一次绊脚的后果。如果策略频率提到 100 Hz 而 γ 不变，视野就缩到 1 s。要保持 2 s，需要 $\gamma' = 1 - \Delta t' / T_{\mathrm{eff}} = 1 - 0.01/2 = 0.995$。等价的换算是 $\gamma' = \gamma^{\Delta t' / \Delta t} = 0.99^{0.5} \approx 0.995$。

> [!robot] **奖励要乘 dt。** legged_gym 把每一项奖励系数乘以策略周期 dt，这样换控制频率时，每秒累积的奖励量级不变。改控制频率时，奖励缩放、γ、episode 步数这三样要一起检查。

> [!robot] **MDP 的"一步"是一个控制周期，而且状态是你定义的。** 一步是 20 ms 的策略周期，不是 5 ms 的仿真步。观测里放什么，决定了问题离马尔可夫有多远。部署时观测必须与训练**逐维对齐**：关节顺序一致（不同 SDK 的腿和关节顺序经常不同）、单位一致、缩放和偏移一致（关节位置要减去同一个默认姿态）、坐标系一致。这一步错了，策略不会报错，只会"走得很奇怪"。

## 常见坑

> [!pitfall] **把仿真状态当观测**：训练时随手把仿真里的真实线速度、足端接触力、地形高度塞进 actor 的观测，仿真里效果惊艳，上真机立刻失效，因为真机根本没有这些量，或者只有很差的估计。规则是：actor 只吃部署时能以相同质量拿到的信号，训练时还要给它们加上与真机相当的噪声；特权信息留给 critic 或 teacher。

> [!pitfall] **奖励不是观测的函数**：奖励用到了 agent 看不到的量，例如有动作变化率惩罚却不给上一步动作，或者按步态相位给奖励却不给相位/时钟输入。对策略来说，同样的观测可能对应不同的奖励，值函数学不准，行为也会出现说不清的抖动。检查方法：逐项列出每个奖励依赖哪些变量，再问一句"这些都在观测里吗？"

> [!pitfall] **超时当终止**：episode 在 20 s 被截断时，如果当作真正的终止处理（未来回报记为 0），值函数就会学到"快到 20 s 时未来没什么价值"。但它看不到时间，于是同一个状态上出现互相矛盾的价值目标，噪声变大。正确做法：真终止（摔倒）不 bootstrap，超时用 $\gamma V(s_{T+1})$ bootstrap。实现细节见 [PPO](#/learn/rl/ppo)。

> [!pitfall] **负奖励诱导"自杀"**：如果训练初期惩罚项占主导、每步总奖励为负，而摔倒会提前结束 episode，那么"尽快摔倒"就是回报最高的策略。legged_gym 的 `only_positive_rewards` 把每步总奖励截到不小于 0，就是为了防这个。终止条件会改变回报的求和范围，它本身就是奖励设计的一部分。

## 自测

> [!quiz] 一个 100 Hz 的策略，希望有效视野约 2 秒，γ 应取多少？如果直接沿用 0.99 会怎样？
> 有效视野步数 $1/(1-\gamma) = 2\ \mathrm{s} \times 100\ \mathrm{Hz} = 200$ 步，所以 $\gamma = 0.995$（也可以按 $0.99^{0.5} \approx 0.995$ 换算）。沿用 0.99，视野只有 1 s，策略对 1 s 之后的后果（比如加速过猛导致的失稳）不敏感，行为更短视。但 γ 越大，回报方差越大、值函数越难学，所以也不是越大越好。

> [!quiz] 只给关节位置、不给关节速度的观测是马尔可夫的吗？给了"位置 + 速度"之后呢？
> 只给位置不是：刚体动力学是二阶的，同样的位置配上不同的速度，下一时刻完全不同。加上速度后，对"理想刚体 + 理想力矩源"可以近似看作马尔可夫。但真机还有执行器动态（电流环、PD 目标、延迟），以及接触、地形、摩擦等不可观测量，所以仍然是 POMDP。工程上靠上一步动作、短历史堆叠或 RNN 来补。

> [!quiz] 为什么训练用高斯随机策略，部署时却用均值？两者不一样会不会有问题？
> 训练需要随机性来探索，也需要可微的 log 概率来算策略梯度；部署时噪声只会带来抖动和额外能耗，所以用均值。问题在于，训练回报是在"带噪声"的策略下评估的，均值策略严格说是另一个策略。训练后期 σ 一般已经很小，两者差别不大；但如果 σ 还很大就部署均值，训练表现和部署表现可能对不上。这也是要监控 action std 的原因之一。

> [!quiz] 机器人摔倒（躯干触地）和 20 s 超时，在计算价值目标时分别怎么处理？为什么不同？
> 摔倒是 MDP 真正的终止状态，之后没有奖励，价值目标里取 $V(s_{T+1}) = 0$。超时只是为了训练方便的人为截断，环境本可以继续，真实的未来回报不为零，应该用 $\gamma V(s_{T+1})$ 估计并 bootstrap。如果把超时当终止，同一个状态（策略看不到时间）会得到两种矛盾的价值目标，引入偏差。

> [!quiz] 观测里的"投影重力"比直接给欧拉角 (roll, pitch, yaw) 好在哪？
> 一是不含 yaw：行走的物理与朝向无关，去掉 yaw 让策略天然对朝向不变，也不受 yaw 漂移影响。二是没有欧拉角的 ±π 跳变和万向锁，信号连续光滑。三是它是单位向量，量纲统一，天然归一化。四是真机上 roll/pitch 有重力作参考，IMU 估计得很可靠，比 yaw 可靠得多。

## 延伸阅读

- [Sutton & Barto, Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — 第 3 章是 MDP、回报、episodic/continuing 最标准的讲法。
- [Spinning Up: Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — 一页讲清策略、轨迹、回报、值函数，符号和本站基本一致。
- [Berkeley CS285](https://rail.eecs.berkeley.edu/deeprlcourse/) — 课程前面的 RL 导论部分系统推导了 MDP、POMDP 与 RL 目标。
- [legged_gym](https://github.com/leggedrobotics/legged_gym) — 读 `legged_robot.py` 里的 `compute_observations`、`check_termination` 和各个 `_reward_*` 函数，就是上面那张表的源码版。

> [!mine] 我的笔记：学完后在这里写下自己的理解、疑问和实验记录。
