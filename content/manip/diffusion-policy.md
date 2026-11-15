> [!one] Diffusion Policy 不直接回归"一个动作"。它用扩散模型表示条件分布 $p(A_t \mid O_t)$：从高斯噪声出发，一步步去噪出一整段未来动作序列。这样它天然能表达多峰（multimodal）的示范，再配上 action chunking 和 receding horizon 执行，就成了把行为克隆在真机上做稳做准的标准配方。

## 直觉

[行为克隆（Behavior Cloning, BC）](#/learn/manip/bc)最朴素的做法是监督回归：网络 $f_\theta(o)$ 输出动作，用 MSE 去拟合示范 $a$。问题在于 **MSE 的最优解是条件均值**：

$$
f^*(o) = \arg\min_f \mathbb{E}\big[\|a - f(o)\|^2\big] = \mathbb{E}[a \mid o]
$$

**Push-T 的例子**：桌上有一个 T 形块，末端执行器（end-effector）要把它推进目标区域。T 块正挡在前面时，示范者有时从左边绕，有时从右边绕，两种都完全合理。假设各占一半，$a_L$ 和 $a_R$ 的均值就是 $(a_L + a_R)/2$，也就是"径直撞上去"。**两个合法 mode 的平均值是一个不合法的动作。** 这就是多峰性问题（multimodality）。人类示范里到处都是这种情况：绕障碍走哪边、先抓哪个物体、手腕往哪个方向转。

即使模型能表示多峰，还有第二个问题：**mode 在时间上来回跳**。第 $t$ 步采样到"左"，第 $t+1$ 步采样到"右"，机器人就会在两种策略之间抖动。解决办法是一次预测一整段动作（action sequence），让整段轨迹属于同一个 mode。

类比图像生成：让模型"画一只猫"，合法的猫有无数种，所有猫的平均是一团灰色的模糊。扩散模型能画出清晰的猫，靠的是它学到了整个分布，而不只是一个均值。Diffusion Policy（Chi et al., RSS 2023）把这个思路搬到动作上：把动作序列当作"图像"来生成，把观测当作"prompt"。

## 形式化：从 DDPM 到动作扩散

### DDPM 回顾

（更完整的推导见[生成模型基础](#/learn/found/generative)。）**前向加噪（forward process）**：对干净样本 $x^0$ 逐步加高斯噪声，

$$
q(x^k \mid x^{k-1}) = \mathcal{N}\big(x^k;\ \sqrt{1-\beta_k}\,x^{k-1},\ \beta_k I\big)
$$

它有闭式解，可以一步跳到任意噪声等级 $k$：

$$
x^k = \sqrt{\bar\alpha_k}\,x^0 + \sqrt{1-\bar\alpha_k}\,\epsilon,\qquad \epsilon \sim \mathcal{N}(0, I),\qquad \bar\alpha_k = \prod_{i=1}^{k}(1-\beta_i)
$$

**训练**：让网络 $\epsilon_\theta$ 从 $x^k$ 中猜出加进去的噪声，

$$
\mathcal{L} = \mathbb{E}_{x^0,\,\epsilon,\,k}\Big[\big\|\epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_k}\,x^0 + \sqrt{1-\bar\alpha_k}\,\epsilon,\ k\big)\big\|^2\Big]
$$

**反向去噪（reverse process）**：从 $x^K \sim \mathcal{N}(0, I)$ 开始迭代，其中 $\alpha_k = 1 - \beta_k$，

$$
x^{k-1} = \frac{1}{\sqrt{\alpha_k}}\Big(x^k - \frac{\beta_k}{\sqrt{1-\bar\alpha_k}}\,\epsilon_\theta(x^k, k)\Big) + \sigma_k z,\qquad z \sim \mathcal{N}(0, I)
$$

关键直觉：噪声预测本质上是 **score function** 的估计，$\epsilon_\theta(x^k, k) \approx -\sqrt{1-\bar\alpha_k}\,\nabla_{x^k}\log q(x^k)$。去噪的每一步都是沿着"对数概率上升的方向"走一小步再加点噪声，相当于一种带退火的 Langevin dynamics。从不同噪声出发会落到不同的 mode，多峰性就这样来了。

### 换成动作：条件去噪

Diffusion Policy 做了两处替换：样本 $x$ 换成**未来一段动作** $A_t \in \mathbb{R}^{T_p \times d_a}$；去噪网络额外以**观测** $O_t$ 为条件。论文的写法是：

$$
A_t^{k-1} = \alpha\Big(A_t^k - \gamma\,\epsilon_\theta(O_t, A_t^k, k) + \mathcal{N}(0, \sigma^2 I)\Big)
$$

$$
\mathcal{L} = \mathrm{MSE}\big(\epsilon^k,\ \epsilon_\theta(O_t,\ A_t^0 + \epsilon^k,\ k)\big)
$$

这里 $\alpha, \gamma, \sigma$ 都由 noise schedule 决定（对照上面的 DDPM 公式：$\alpha = 1/\sqrt{\alpha_k}$，$\gamma = \beta_k/\sqrt{1-\bar\alpha_k}$）。$A_t^0 + \epsilon^k$ 是"按第 $k$ 级加噪"的简写。论文实验发现 iDDPM 的 **square cosine schedule** 效果最好。

**为什么是 $p(A_t \mid O_t)$，而不是联合分布 $p(A_t, O_t)$？** 之前的 Diffuser（Janner et al.）对状态和动作的联合轨迹做扩散，靠 inpainting 来施加条件。Diffusion Policy 只建模条件分布，观测只作为条件输入，有两个好处：

1. **视觉特征每次推理只算一次**，不用在 $K$ 步去噪里反复处理图像，速度快很多；
2. 视觉编码器可以**端到端**跟着去噪 loss 一起训练，不需要预测未来图像。

### 和 Implicit BC 的关系

Implicit BC（IBC）用能量模型（Energy-Based Model, EBM）表示策略：

$$
p_\theta(a \mid o) = \frac{\exp(-E_\theta(o, a))}{Z(o, \theta)}
$$

训练时要用负样本去估计归一化常数 $Z$（InfoNCE 式的 loss），这正是 IBC 训练不稳定的根源。扩散模型学的是 $\nabla_a \log p(a \mid o) = -\nabla_a E_\theta(o, a) - \nabla_a \log Z(o, \theta)$，而 $Z$ 与 $a$ 无关，所以 $\nabla_a \log Z = 0$。**训练和推理都不需要 $Z$**，loss 退化成一个普通的 MSE，非常稳定。可以把 Diffusion Policy 理解为"学梯度场、而不是学能量本身"的 IBC。

## 方法：架构与执行方式

### Action chunking 与 receding horizon

三个时间窗口：

| 符号 | 含义 | 典型值 |
| --- | --- | --- |
| $T_o$ | observation horizon：输入最近几帧观测 | 2 |
| $T_p$ | prediction horizon：一次去噪出多少步动作 | 16 |
| $T_a$ | action execution horizon：实际执行多少步再重新规划 | 8 |

执行方式类似 MPC：拿最近 $T_o$ 帧观测，去噪出 $T_p$ 步动作，**只执行前 $T_a$ 步**，然后用新观测重新预测。$T_a$ 是在**时序一致性**和**响应速度**之间取折中：$T_a$ 太小，chunk 之间频繁切 mode，又开始抖；$T_a$ 太大，就接近开环执行，对扰动反应慢。论文发现 $T_a = 8$ 对大多数任务最好。

动作序列还带来一个副作用：**对示范里的停顿（idle actions）更鲁棒**。比如倒酱汁时人会停一下，单步策略在静止状态下容易"卡住不动"，预测序列则能把停顿之后的动作也一起规划出来。

```python
# 训练（一个 batch）
obs_feat = encoder(obs[:, :T_o])                 # 视觉 + 本体特征，只算一次
k = torch.randint(0, K, (B,))
noise = torch.randn_like(actions)                # actions: (B, T_p, d_a)，已归一化到 [-1, 1]
noisy = scheduler.add_noise(actions, noise, k)
loss = F.mse_loss(eps_net(noisy, k, cond=obs_feat), noise)

# 推理（receding horizon）
A = torch.randn(1, T_p, d_a)
for k in scheduler.timesteps:                    # DDIM 下 ~10 步
    A = scheduler.step(eps_net(A, k, cond=obs_feat), k, A)
robot.execute(unnormalize(A)[0, :T_a])           # 只执行前 T_a 步，然后重新观测、重新预测
```

### 两种去噪骨干

（Transformer 部分的背景见 [Transformer](#/learn/found/transformer)。）

| | CNN-based | Transformer-based |
| --- | --- | --- |
| 结构 | 1D temporal U-Net（沿用 Diffuser），在时间维上卷积 | minGPT 风格的 decoder：带噪动作作为 token，观测 embedding 通过 cross-attention 注入，动作 token 之间用 causal attention |
| 条件注入 | **FiLM**：观测特征和扩散步 $k$ 生成逐通道的 scale/bias | 扩散步 $k$ 作为额外 token，观测走 cross-attention |
| 优点 | 开箱即用，对超参不敏感 | 动作变化剧烈、高频时更好（如速度控制） |
| 缺点 | 卷积偏好低频信号，动作需要突变时表现差 | 对超参更敏感，需要调 |
| 建议 | **先试这个** | CNN 不够时再换 |

**视觉编码器**：每个相机一个 ResNet-18，不用预训练权重，端到端训练。改了两处：global average pooling 换成 **spatial softmax**（保留空间位置信息），BatchNorm 换成 **GroupNorm**（BatchNorm 和 EMA 权重一起用会出问题）。本体状态（proprioception，比如末端位姿和夹爪开度）直接拼接到观测特征里。

### 少步推理：DDIM

训练用 100 步扩散。推理时用 DDIM 把采样和训练步数解耦：先由当前噪声估计干净样本，再直接跳到更低的噪声等级 $k' < k$（取 $\eta = 0$ 时是确定性采样）：

$$
\hat{x}^0 = \frac{x^k - \sqrt{1-\bar\alpha_k}\,\epsilon_\theta(x^k, k)}{\sqrt{\bar\alpha_k}},\qquad x^{k'} = \sqrt{\bar\alpha_{k'}}\,\hat{x}^0 + \sqrt{1-\bar\alpha_{k'}}\,\epsilon_\theta(x^k, k)
$$

论文的真机设置是 100 步训练、10 步 DDIM 推理，在 RTX 3080 上延迟约 0.1 s。

### 位置控制 vs 速度控制

一个反直觉的发现：多数 baseline（如 LSTM-GMM、BeT）用速度或增量控制（velocity / delta）更好，**Diffusion Policy 却是绝对位置控制（position control）稳定更好**。论文给的两个原因：

1. 位置空间里的多峰性更明显，恰好是扩散模型擅长、其他方法吃亏的地方；
2. 位置控制的复合误差（compounding error）更小，更适合预测动作序列。速度控制对延迟也更敏感。

### 与其他 BC 方法的对比

| 方法 | 动作分布怎么表示 | 多峰 | 训练稳定性 | 推理开销 | 输出 |
| --- | --- | --- | --- | --- | --- |
| MSE-BC | 单点（条件均值） | 否 | 最稳 | 1 次前向 | 单步或 chunk |
| LSTM-GMM（BC-RNN） | 高斯混合，$K$ 个分量 | 有限，分量数要调，高维下难训 | 较稳 | 1 次前向 | 单步 |
| Implicit BC（EBM） | 能量函数 $E_\theta(o, a)$ | 是 | 需负样本估 $Z$，不稳 | 推理要采样或优化，慢 | 单步 |
| BeT | k-means 离散 bin + 连续 offset | 是（bin 级） | 稳 | 1 次前向 | 单步（带历史） |
| [ACT](#/learn/manip/act)（CVAE） | 潜变量 $z$ + transformer 解码 chunk | 训练时建模风格；推理时 $z$ 取先验均值，基本是确定性的 | 稳，需调 KL 权重 | 1 次前向 | chunk + temporal ensembling |
| **Diffusion Policy** | score / 噪声预测，迭代去噪 | **是，而且是序列级多峰** | 稳（噪声 MSE） | $K$ 次去噪（DDIM 约 10） | chunk + receding horizon |

论文在 4 个 benchmark（Robomimic、Push-T、Block Pushing、Franka Kitchen）的十余个任务上，相对此前最好的方法平均提升 46.9%。

### 后续发展

- **3D Diffusion Policy（DP3）**：把输入换成稀疏点云，用一个轻量的点云编码器提特征。论文报告大部分仿真任务只要 10 条示范，真机任务每个 40 条，对视角和外观变化的泛化更好。
- **Consistency Policy**：把训练好的 Diffusion Policy 蒸馏成 consistency model，一到几步就能出动作，适合算力有限的机器人。
- **Flow matching 动作头**：[π0](#/learn/vla/pi0) 的 action expert 用的是 flow matching，不是 DDPM。构造 $A^\tau = \tau A + (1-\tau)\epsilon$，回归速度场 $v_\theta(A^\tau, o) \approx A - \epsilon$，推理时从 $\tau = 0$ 的噪声开始用 Euler 积分到 $\tau = 1$（π0 用 10 步，chunk 长度 50）。路径更直，少步采样更友好，现在是 [VLA](#/learn/vla/overview) 连续动作头的主流选择。
- **一个反例值得记住**：OpenVLA-OFT 发现，在 LIBERO 上用 chunking + 并行解码 + **L1 回归**，效果和速度都很好。多峰性到底有多重要取决于数据：示范越一致、越单峰，回归就越够用；示范来自多人、多策略时，生成式动作头的优势才明显。

## 机器人视角

> [!robot] **动作归一化到 [-1, 1]**：扩散的噪声是单位高斯，去噪结果通常也会被 clip 到 [-1, 1]，所以每个动作维度都要做 min-max 归一化（官方实现就是这么做的）。用绝对位姿时，旋转不要用欧拉角（会有 ±π 跳变），用 6D rotation 这类连续表示。

> [!robot] **EMA 权重**：评测和部署用 EMA 权重，不要用最后一步的原始权重，差别往往很明显。这也是视觉编码器要用 GroupNorm 的原因：BatchNorm 的 running statistics 和 EMA 配合得不好。

> [!robot] **延迟预算**：真机 Push-T 里，策略以 10 Hz 预测，动作插值到 125 Hz 下发给控制器。动作序列天然能"盖住"推理延迟：执行上一段 chunk 的同时在算下一段，并把时间戳已经过期的动作丢掉。观测和动作的**时间戳对齐**比模型结构更重要。10 步 DDIM 对小模型够快；模型一大（比如 VLA），就要考虑 flow matching、蒸馏，或者异步推理。

> [!robot] **数据的质量比数量更重要**：Diffusion Policy 能"容忍"多峰，但不能容忍乱七八糟的示范。同一个人、一致的策略、流畅的[遥操作](#/learn/manip/teleop)，几十到一两百条示范就能训出一个单任务策略；数据里如果混入大量犹豫、回退和失败恢复，策略也会学会这些动作。评测要做足够多的真机 rollout（见[评测](#/learn/manip/eval)），不要只看几次成功的录像。

> [!robot] **从哪里开始**：[LeRobot](https://github.com/huggingface/lerobot) 里有 Diffusion Policy 的实现和 Push-T 环境，官方 repo 有完整的训练配置。建议先在 Push-T 上把 $T_o, T_p, T_a$ 和 DDIM 步数各调一遍，体会每个超参的作用。

## 常见坑

> [!pitfall] **看 loss 选 checkpoint**：扩散 loss 是噪声预测误差，它的下降和任务成功率相关性很弱，训练后期 loss 几乎不动，策略却可能还在变好或变差。正确做法是定期做 rollout 评测（仿真）或固定一组真机测试，用成功率选 checkpoint。

> [!pitfall] **归一化统计量不一致**：训练用一套 min/max，部署时忘了反归一化，或者换了数据之后没有重算统计量，机器人就会以错误的尺度乱动。归一化参数要和模型权重一起保存，部署脚本直接加载。夹爪这类近似二值的维度也要单独检查。

> [!pitfall] **$T_a$ 设得太大**：为了省推理次数把 $T_a$ 设成 $T_p$，等于开环执行一整段 chunk，遇到扰动或物体滑动反应不过来；$T_a = 1$ 又会重新引入 mode 抖动。先用 $T_a \approx T_p / 2$，再按任务的动态程度调整。

> [!pitfall] **观测历史太长**：直觉上更多历史帧应该更好，实际上 $T_o$ 变大常常让策略更容易过拟合到虚假的时间相关性（causal confusion），推理也更慢。$T_o = 2$（能看出速度信息）通常就够了。

## 自测

> [!quiz] 用 MSE 回归一个双峰动作分布，最优解是什么？为什么在机器人上这比在图像上更致命？
> MSE 的最优解是条件均值 $\mathbb{E}[a \mid o]$。若示范一半是 $a_L$、一半是 $a_R$，网络会输出 $(a_L + a_R)/2$。图像的均值只是模糊；动作的均值可能直接落在两个合法 mode 之间的不可行区域，比如撞上障碍或者抓空。而且是闭环执行：一步错了，后面的状态都偏离示范分布（compounding error），越走越远。

> [!quiz] Diffusion Policy 为什么建模 $p(A_t \mid O_t)$ 而不是像 Diffuser 那样建模 $p(A_t, O_t)$？
> 一是**计算**：观测只作为条件，视觉编码器每次推理只前向一次，$K$ 步去噪只跑轻量的去噪网络；建模联合分布则要在每步去噪时连同观测或未来观测一起处理。二是**学习目标**：策略只需要动作，没必要去生成未来图像或状态，这样容量全部用在动作上，视觉编码器也能端到端地只为动作预测服务。

> [!quiz] Implicit BC 和 Diffusion Policy 都能表达多峰，为什么前者训练不稳定、后者稳定？
> IBC 直接学能量 $E_\theta(o, a)$，最大似然需要归一化常数 $Z(o, \theta)$，只能用负样本近似（InfoNCE），近似的质量取决于负样本采样，在高维动作空间里很差。扩散模型学的是 score $\nabla_a \log p(a \mid o)$，而 $\nabla_a \log Z = 0$，所以完全绕开了 $Z$。训练目标是对噪声的 MSE 回归，监督信号稠密、方差小。

> [!quiz] 如果把 $T_a$ 从 8 改成 1 或 16（$T_p = 16$），分别会发生什么？
> $T_a = 1$：每步都重新采样，相邻两步可能落在不同 mode，动作抖动，在 Push-T 这种"左右都行"的场景会来回犹豫；同时推理频率要求最高。$T_a = 16$：整段开环执行，chunk 内部一致，但对外界变化（物体被碰动、抓取打滑）反应延迟长达 16 个控制周期，而且 chunk 后半段是基于更旧的观测预测的，误差更大。中间值兼顾一致性和闭环反馈。

> [!quiz] 训练用了 100 步扩散，推理为什么可以只走 10 步？还有哪些办法更快？
> DDIM 把采样改写成：先由 $\epsilon_\theta$ 估计 $\hat{x}^0$，再按闭式公式直接跳到任意更低的噪声等级。它和 DDPM 共享同一个训练目标，所以不用重训，只是用更稀疏的时间步子序列去积分，代价是步数过少时质量下降。更快的方法有：flow matching（路径更直，π0 用 10 步 Euler），consistency 蒸馏（Consistency Policy，一到几步），或者干脆像 OpenVLA-OFT 那样在数据足够单峰时直接回归。

## 延伸阅读

- [Diffusion Policy 项目主页](https://diffusion-policy.cs.columbia.edu) — 真机视频很直观，能看到多峰行为和对扰动的鲁棒性
- [论文 arXiv:2303.04137](https://arxiv.org/abs/2303.04137) — 第 4 节的设计取舍（chunking、位置控制、视觉编码器）比主结果表更值得精读
- [官方代码 real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) — 配置文件里有每个任务的完整超参
- [LeRobot](https://github.com/huggingface/lerobot) — 工程上更轻的实现，适合拿来在自己的机器人上跑
- [3D Diffusion Policy（arXiv:2403.03954）](https://arxiv.org/abs/2403.03954) — 点云输入如何提升数据效率和泛化
- [Consistency Policy（arXiv:2405.07503）](https://arxiv.org/abs/2405.07503) — 把扩散策略蒸馏到一到几步推理

> [!mine] 我的笔记：学完后在这里写下自己的理解、疑问和实验记录。
