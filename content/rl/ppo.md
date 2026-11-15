> [!one] PPO 用一个"裁剪过的重要性比率"把每次更新限制在旧策略附近，于是可以放心地在同一批数据上做多轮小批量梯度更新。它简单、稳定，又吃得下海量并行数据，所以成了腿式机器人 RL 的默认算法。

## 直觉

[策略梯度](#/learn/rl/pg) 的思想很朴素：advantage 为正的动作调高概率，为负的调低。难的是"调多少"：

- 策略梯度只是当前策略处的**局部**估计。步子一大，新策略访问的状态分布就变了，原来的估计随之失效。
- RL 没有固定数据集兜底。策略一旦崩坏（比如腿学会了乱踢），之后采到的数据也全是垃圾，**一次坏更新会污染后面所有数据**，很难恢复。
- 对神经网络来说，参数空间里的"一小步"可能是策略空间里的"一大步"，学习率很难调。

所以 vanilla PG 的典型症状是：学习率小了慢得要命，大了回报会断崖下跌。而且每批数据只能做一次梯度更新（更新完数据就变成 off-policy 的了），样本效率很低。

TRPO 和 PPO 回答的是同一个问题：**怎样在同一批数据上尽量多地改进策略，又不走出"估计还可信"的范围？** TRPO 用硬 KL 约束加二阶优化；PPO 用一个裁剪的目标函数，只靠一阶的 Adam 就能近似做到。

## 形式化

### 从策略梯度到替代目标

带 advantage 的策略梯度是：

$$
\nabla_\theta J(\theta) = \mathbb{E}_t\Big[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat{A}_t\Big]
$$

数据由旧策略 $\pi_{\theta_{\mathrm{old}}}$ 采集。定义**重要性比率**（importance ratio）：

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}
$$

再定义替代目标（surrogate objective）$L^{\mathrm{CPI}}(\theta) = \mathbb{E}_t[r_t(\theta)\hat{A}_t]$。在 $\theta = \theta_{\mathrm{old}}$ 处 $r_t = 1$，且 $\nabla_\theta r_t = \nabla_\theta \log \pi_\theta$，所以它的梯度和策略梯度完全一样。不同的是，它是一个可以连续优化好几步的目标。问题在于，无约束地最大化它，$r_t$ 会被推得很远。重要性比率只修正了**动作**分布，状态分布仍然是旧策略的，一旦走远，这个目标就不可信了。

### TRPO：一段话版本

TRPO（Trust Region Policy Optimization）把上面的想法写成带约束的优化：最大化 $\mathbb{E}_t[r_t \hat{A}_t]$，同时约束平均 $\mathrm{KL}(\pi_{\theta_{\mathrm{old}}} \Vert \pi_\theta) \le \delta$。它的理论结果是：真实目标的改进不小于替代目标的改进减去一个与最大 KL 成正比的惩罚，所以在信赖域（trust region）内优化替代目标可以保证单调改进（实践中用平均 KL 近似）。实现上，它对 KL 做二阶近似（Fisher 矩阵），用共轭梯度求自然梯度方向，再线搜索确保约束满足。效果稳，但实现复杂，每步计算也贵，和共享参数的网络结构配合起来也麻烦。PPO 的出发点就是用一阶方法拿到 TRPO 的大部分好处。

### 裁剪替代目标

$$
L^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(r_t(\theta)\hat{A}_t, \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\big)\Big]
$$

按 advantage 的符号分两种情况看（下面取 $\epsilon = 0.2$）。

**情况 1：$\hat{A}_t > 0$**，这个动作比平均好，要提高它的概率。把 $\hat{A}_t$ 提出来，目标变成 $\hat{A}_t \cdot \min(r_t, \operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)) = \hat{A}_t \cdot \min(r_t, 1+\epsilon)$：

- $r_t < 1+\epsilon$：目标等于 $r_t\hat{A}_t$，梯度正常，概率继续提高；
- $r_t > 1+\epsilon$：目标恒为 $(1+\epsilon)\hat{A}_t$，**梯度为 0**，表示已经提得够多了，别再贪；
- 注意 $r_t < 1-\epsilon$ 时**不裁剪**（概率被其他样本的更新带着降低了）：目标仍是 $r_t\hat{A}_t$，梯度会把它拉回来。

**情况 2：$\hat{A}_t < 0$**，这个动作比平均差，要降低它的概率。乘以负数后 min 变成 max，目标是 $\hat{A}_t \cdot \max(r_t, 1-\epsilon)$：

- $r_t > 1-\epsilon$：梯度正常，概率继续降低；
- $r_t < 1-\epsilon$：目标恒为 $(1-\epsilon)\hat{A}_t$，梯度为 0，已经降够了；
- $r_t > 1+\epsilon$ 时**不裁剪**（坏动作的概率反而变大了），梯度全力纠正。

一句话总结：**裁剪只拿掉"让目标显得更好"的那部分改进，从不拿掉"让目标变差"的惩罚**。所以 $L^{\mathrm{CLIP}}$ 是未裁剪目标的一个悲观下界（pessimistic bound）。

还要注意，比率越界的样本只是"不再继续推"，PPO 并没有硬性保证 KL 很小。epoch 多、学习率大时，策略照样可能走远。这就是实践中还要监控 KL、用 KL 自适应调学习率的原因。

### 价值损失与熵奖励

PPO 用 actor-critic 结构（见 [Actor-Critic](#/learn/rl/ac)），同时训练价值网络 $V_\phi$。它的回归目标是回报估计 $\hat{R}_t = \hat{A}_t + V_{\phi_{\mathrm{old}}}(s_t)$：

$$
L^{V}(\phi) = \mathbb{E}_t\Big[\big(V_\phi(s_t) - \hat{R}_t\big)^2\Big]
$$

rsl_rl 默认还对价值做了类似的裁剪：取 $\max\big((V_\phi - \hat{R})^2, (V^{\mathrm{clip}} - \hat{R})^2\big)$，其中 $V^{\mathrm{clip}} = V_{\mathrm{old}} + \operatorname{clip}(V_\phi - V_{\mathrm{old}}, -\epsilon, \epsilon)$。

熵奖励（entropy bonus）用来鼓励持续探索。对角高斯的熵只和 σ 有关：$H = \sum_i \big(\log \sigma_i + \tfrac{1}{2}\log(2\pi e)\big)$。要最小化的总损失是：

$$
L(\theta, \phi) = -L^{\mathrm{CLIP}}(\theta) + c_1 L^{V}(\phi) - c_2 \mathbb{E}_t\big[H[\pi_\theta(\cdot \mid s_t)]\big]
$$

### GAE

advantage 用 GAE（Generalized Advantage Estimation）估计。先定义 TD 残差：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

再做指数加权求和，并写成便于倒序计算的递推形式（$d_t$ 是 episode 结束标志）：

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \qquad \hat{A}_t = \delta_t + \gamma\lambda(1 - d_t)\hat{A}_{t+1}
$$

λ 的直觉：

- $\lambda = 0$：$\hat{A}_t = \delta_t$，即一步 TD。方差小，但完全依赖 V 的准确性，V 不准时偏差大。
- $\lambda = 1$：求和逐项抵消，得到 $\hat{A}_t = \sum_l \gamma^l r_{t+l} - V(s_t)$，即蒙特卡洛回报减基线。相对折扣目标无偏，但方差大。
- 中间值在各种 n 步回报之间插值。$\gamma\lambda = 0.99 \times 0.95 \approx 0.94$，权重 $(\gamma\lambda)^l$ 的有效长度约 $1/(1-\gamma\lambda) \approx 17$ 步。可以理解为 GAE 大约"真实地看" 17 步奖励，更远的部分交给 V 去估。

## 方法

### 完整训练循环

```python
# N 个并行环境，每轮每个环境采 T 步：batch = N * T
obs = envs.reset()                                   # [N, obs_dim]
for it in range(max_iterations):
    # ---------- 1. Rollout（不求梯度）----------
    buf = RolloutBuffer(T, N)
    for t in range(T):
        with torch.no_grad():
            mu, std = actor(normalize(obs))          # 高斯策略
            a = mu + std * torch.randn_like(mu)
            logp = gaussian_logprob(a, mu, std)      # 各动作维求和
            v = critic(normalize(obs))
        next_obs, r, done, timeout = envs.step(a)    # done 包含 timeout；环境会自动 reset
        r = r + gamma * v * timeout                  # 超时 bootstrap（rsl_rl 做法：用 V(s_t) 近似）
        buf.add(obs, a, logp, mu, std, v, r, done)
        obs = next_obs
    with torch.no_grad():
        last_v = critic(normalize(obs))

    # ---------- 2. GAE（倒序计算）----------
    adv = torch.zeros(T, N); gae = 0
    for t in reversed(range(T)):
        next_v = last_v if t == T - 1 else buf.v[t + 1]
        nonterminal = 1.0 - buf.done[t]
        delta = buf.r[t] + gamma * next_v * nonterminal - buf.v[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    ret = adv + buf.v                                # value 的回归目标（在归一化之前算）
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)    # advantage 归一化

    # ---------- 3. 多轮小批量更新 ----------
    for epoch in range(num_epochs):                  # 如 5
        for mb in buf.shuffled_minibatches(num_minibatches, adv, ret):   # 如 4
            mu, std = actor(normalize(mb.obs))
            logp = gaussian_logprob(mb.a, mu, std)
            ratio = torch.exp(logp - mb.logp_old)
            surr = torch.min(ratio * mb.adv,
                             ratio.clamp(1 - clip, 1 + clip) * mb.adv)
            policy_loss = -surr.mean()

            v = critic(normalize(mb.obs))
            v_clip = mb.v_old + (v - mb.v_old).clamp(-clip, clip)
            value_loss = torch.max((v - mb.ret) ** 2, (v_clip - mb.ret) ** 2).mean()

            entropy = gaussian_entropy(std).mean()
            loss = policy_loss + c1 * value_loss - c2 * entropy

            # 按 KL(old || new) 自适应调学习率
            kl = gaussian_kl(mb.mu_old, mb.std_old, mu, std).mean()
            if kl > 2.0 * desired_kl:
                lr = max(1e-5, lr / 1.5)
            elif kl < desired_kl / 2.0:
                lr = min(1e-2, lr * 1.5)
            set_lr(optimizer, lr)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)   # 如 1.0
            optimizer.step()
```

超时那一行值得多说一句。并行环境在 done 时会自动 reset，`next_obs` 已经是新 episode 的观测，拿不到真正的 $s_{T+1}$。rsl_rl 的近似办法是：在超时这一步的奖励上加 $\gamma V(s_T)$，再按普通的 done 截断 GAE。

### 典型超参数

下表是 rsl_rl / legged_gym 默认配置风格的典型值：

| 超参数 | 典型值 | 说明 |
| --- | --- | --- |
| 并行环境数 N | 4096 | GPU 仿真，见 [大规模并行仿真](#/learn/sim/parallel) |
| 每环境步数 T | 24 | 50 Hz 下约 0.5 s |
| batch | 4096 × 24 = 98304 | 一轮约 33 分钟的"机器人经验" |
| learning epochs | 5 | |
| minibatches | 4 | 每个约 2.5 万样本，每轮共 20 次梯度步 |
| clip ε | 0.2 | |
| γ / λ | 0.99 / 0.95 | |
| learning rate | 1e-3，adaptive schedule，desired KL 0.01 | |
| value loss 系数 $c_1$ | 1.0 | 开启 value clipping |
| entropy 系数 $c_2$ | 0.01 | |
| max grad norm | 1.0 | |
| 初始动作噪声 std | 1.0 | 可学习、与状态无关 |
| 网络 | actor / critic 各一个 MLP [512, 256, 128]，ELU | |

这些只是一个"能跑起来"的起点，不是金科玉律。换机器人、换任务（尤其是难地形或人形）后，epoch 数、std 初值、entropy 系数经常要改。

### 三个归一化与调度技巧

1. **观测归一化**。各个观测的量级可以差好几个数量级：关节速度在 10 rad/s 量级，投影重力在 1 量级。legged_gym 用手工缩放（例如关节速度乘 0.05、角速度乘 0.25），再裁剪到 ±100；新版 rsl_rl / Isaac Lab 也提供 running mean/std 的经验归一化。关键是**部署时要用同一套缩放或统计量**，经验归一化的均值、方差必须和网络权重一起导出。
2. **Advantage 归一化**。把整个 batch（或每个 minibatch）的 $\hat{A}$ 减去均值、除以标准差。这样策略损失的量级与奖励尺度无关，学习率更好调。副作用是总有大约一半样本被"压低"，哪怕所有动作其实都不错。它做的是相对比较，不是绝对评价。
3. **按 KL 自适应学习率**。每个 minibatch 都计算新旧高斯策略之间的 KL：

$$
\mathrm{KL}(\mathcal{N}_{\mathrm{old}} \Vert \mathcal{N}_{\mathrm{new}}) = \sum_i \Big[\log\frac{\sigma_i^{\mathrm{new}}}{\sigma_i^{\mathrm{old}}} + \frac{(\sigma_i^{\mathrm{old}})^2 + (\mu_i^{\mathrm{old}} - \mu_i^{\mathrm{new}})^2}{2(\sigma_i^{\mathrm{new}})^2} - \frac{1}{2}\Big]
$$

超过目标的 2 倍就把 lr 除以 1.5，低于目标的一半就乘以 1.5，并限制在 [1e-5, 1e-2] 内。这相当于一个廉价的 trust region：比固定 lr 稳得多，也省了调 lr 的力气。

### 为什么 PPO 和 GPU 大规模并行是绝配

- **On-policy 的短板被抹平**。PPO 每批数据用完就扔，样本效率低；但 GPU 仿真（Isaac Gym / Isaac Lab 等）每秒能产生大量数据，样本不再稀缺，**墙钟时间**才是瓶颈。
- **大 batch 天然降低方差**。约 10 万样本的 batch 让梯度估计非常平滑，进一步放大了 PPO 的稳定性。
- **不需要 replay buffer**。数据就在 GPU 上，采完即用，没有大缓冲区的显存开销，也没有 CPU–GPU 拷贝。
- **简单、鲁棒、对 reward 改动不敏感**。腿式 RL 的主要工作量在 reward、观测、域随机化和课程上，需要一个不用调就能跑的优化器，好让注意力留在任务设计上。改完 reward 不用重调十个超参数，工程上非常值钱。
- **T 可以很短**。有 V 做 bootstrap，每个环境只需采 24 步，远短于 1000 步的 episode。4096 个环境处在 episode 的不同阶段，合起来覆盖了整个状态分布。

相比之下，SAC 等 off-policy 方法（见 [SAC](#/learn/rl/sac)）样本效率高，在真机上直接学习、或数据昂贵的场景里更有优势。近年也有把 off-policy 方法扩展到大规模并行仿真的工作，但在腿式运动上，PPO 仍是默认基线。

## 机器人视角

> [!robot] **一轮迭代值多少经验？** 4096 个环境 × 24 步 × 0.02 s ≈ 1966 s，约 33 分钟的机器人经验，而在 GPU 上这只是一轮迭代。1500 轮迭代下来约为 34 天不间断的经验。这就是"在仿真里几分钟到几小时学会走路"的底气，也说明了为什么 on-policy 的低样本效率在这里不是问题。

> [!robot] **读训练日志的顺序。** 先看平均回报和 episode 长度（有没有摔倒）；再看 value loss（是否在下降、量级是否合理）；再看 rsl_rl 日志里的 mean noise std（探索有没有在合理收缩）；最后看 KL 和学习率（自适应调度是不是一直顶在上限或下限）。各项奖励分量要分开记录：总回报涨了，可能只是某个惩罚项变小了，并不是走得更好。

> [!robot] **actor 和 critic 可以吃不同的观测。** 在仿真里，critic 可以用特权观测（真实线速度、地形高度、摩擦），actor 只用真机能拿到的观测。非对称 actor-critic 在 PPO 里几乎零成本，是缓解部分可观测导致 value 学不好的标准做法（见 [MDP](#/learn/rl/mdp)、[Teacher-Student](#/learn/loco/teacher)）。

## 常见坑

> [!pitfall] **Value function 学不动**：症状是 value loss 很大或者不降，回报曲线很噪。常见原因有四个：奖励尺度太大（回报量级约为 $r/(1-\gamma)$，单步奖励 1 对应回报 100）；观测没归一化；critic 看不到决定回报的关键信息（部分可观测）；奖励尺度大时 value clipping 反而拖慢了 V 的跟进。对策：缩放奖励（legged_gym 把每项奖励乘 dt）、归一化观测、给 critic 特权观测，并且经常看一眼 value loss 的数值量级。

> [!pitfall] **熵崩塌 / action std 爆炸**：std 是可学习参数，两个方向都会出问题。崩塌是 std 过早缩得很小，探索停止，策略卡在局部最优（比如原地踏步）。爆炸是奖励对动作噪声不敏感或 entropy 系数偏大时，std 一路上涨；仿真里噪声被 PD 和力矩限幅"吸收"，看起来没事，部署均值策略时行为却和训练时不一样。对策：监控 mean noise std，调 entropy 系数，必要时给 std 设上下限或改用 log-std 参数化。

> [!pitfall] **奖励尺度与负奖励**：advantage 归一化抵消了一部分尺度问题，但 value 回归和 value clipping 仍然受尺度影响。更隐蔽的是负奖励：如果训练初期每步总奖励为负，提前摔倒反而回报更高，策略就会学会"自杀"。legged_gym 的 `only_positive_rewards` 把每步总奖励截到不小于 0，就是为了防这个。

> [!pitfall] **截断与终止的 bootstrap 混用**：超时不是终止，价值目标应该是 $r_T + \gamma V(s_{T+1})$；摔倒才是终止，目标就是 $r_T$。并行环境自动 reset 导致拿不到 $s_{T+1}$，所以要像上面伪代码那样在超时步的奖励上补 $\gamma V(s_T)$。漏掉这一步，V 在 episode 末尾会被系统性低估，还会依赖策略根本看不到的"时间"。

## 自测

> [!quiz] ε=0.2，某个样本 $\hat{A}_t < 0$，当前 $r_t = 1.5$，它对策略参数的梯度是零吗？如果 $r_t = 0.7$ 呢？
> $r_t = 1.5$ 时，目标为 $\hat{A}_t \max(r_t, 0.8) = 1.5\hat{A}_t$，没有被裁剪，梯度不为零。这个坏动作的概率被错误地推高了，PPO 会全力把它拉回来。$r_t = 0.7 < 0.8$ 时，目标是常数 $0.8\hat{A}_t$，梯度为零，因为概率已经降得够多了。这正是"只裁掉乐观的改进，不裁掉悲观的惩罚"。

> [!quiz] episode 长 20 s（50 Hz 下 1000 步），但每个环境每轮只采 24 步。为什么这样也能学好？
> 原因有三。第一，GAE 用 $V(s_{t+24})$ 对剩余回报做 bootstrap，不需要采到 episode 结束。第二，$\gamma\lambda \approx 0.94$，GAE 的有效长度约 17 步，24 步已经覆盖了大部分权重。第三，4096 个环境处在 episode 的不同阶段，合起来覆盖了完整的状态分布。代价是 advantage 的质量依赖 V 的准确性，V 学得差时短 rollout 偏差大，这时可以适当加大 T。

> [!quiz] 为什么 PPO 不能把上一轮、甚至十轮前的数据存起来复用？
> PPO 是 on-policy 的，替代目标只在 $\theta$ 接近 $\theta_{\mathrm{old}}$ 时可信。重要性比率只修正了动作分布，没修正状态分布；旧数据的 advantage 也是用旧的 V、在旧策略下估出来的。数据越旧，比率离 1 越远，大部分样本要么被裁剪（梯度为零），要么方差爆炸。想复用旧数据，就得用 off-policy 方法，比如 SAC 那样学 Q 函数。

> [!quiz] 开着自适应学习率调度，你发现 lr 一直被压在下限 1e-5，KL 却仍然远超目标。可能是什么问题？
> 这说明哪怕极小的步长也会引起很大的策略变化。最常见的是 action std 塌缩到很小：KL 中 $(\Delta\mu)^2 / 2\sigma^2$ 这一项在 σ 很小时会爆炸，同样的均值变化对应巨大的 KL。其次是观测没归一化，网络输出对参数极其敏感；还可能有数值问题（NaN、极端观测值）。先查 noise std 和观测统计量。

> [!quiz] 把 λ 从 0.95 改成 1.0 或 0.0，训练分别会有什么变化？
> λ=1：advantage 接近蒙特卡洛回报减基线，对 V 的误差不敏感，但方差大，训练曲线更噪。注意 T=24 的短 rollout 在末尾仍要用 $V(s_T)$ bootstrap，所以并不是纯 MC。λ=0：只用一步 TD 残差，方差最小，但强烈依赖 V；训练初期 V 很差，advantage 几乎就是噪声，学得慢甚至学偏。0.95 是两者之间的经验折中。

## 延伸阅读

- [PPO 原论文](https://arxiv.org/abs/1707.06347) — 第 3 节的裁剪目标，加上图 1 里正负 advantage 两种情况的示意图，是本课的核心。
- [GAE 论文](https://arxiv.org/abs/1506.02438) — 把 γ 和 λ 都解释成偏差–方差权衡，推导短而清楚。
- [Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — 伪代码简洁，还给出了裁剪目标按 advantage 符号拆开的简化写法。
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) — 腿式机器人最常用的 PPO 实现。`ppo.py` 和 rollout storage 加起来不长，值得对照本课伪代码逐行读。
- [Berkeley CS285](https://rail.eecs.berkeley.edu/deeprlcourse/) — 策略梯度和 actor-critic 那几讲把从 PG 到 advantage 的来龙去脉讲得很清楚。

> [!mine] 我的笔记：学完后在这里写下自己的理解、疑问和实验记录。
