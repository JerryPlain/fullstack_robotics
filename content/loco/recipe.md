> [!one] 标准配方是：几千个 GPU 并行环境 + PPO + "默认姿态上的关节位置偏移 → PD"的动作接口 + 一组"跟踪为主、惩罚为辅"的奖励 + 地形/指令课程 + 域随机化。能不能上真机，一半取决于部署端能否把观测、动作和时序原样复刻出来。

## 直觉

几年前，用 RL 让四足学会走路足以发一篇顶会；今天它是一个下午的活。主力算法没变，还是 2017 年的 [PPO](#/learn/rl/ppo)。变的是社区收敛出了一套配方：

- **样本靠堆**：GPU 物理仿真（Isaac Gym → Isaac Lab，见 [并行仿真](#/learn/sim/parallel)）让单卡同时跑几千个机器人，on-policy 算法样本效率低的缺点被算力抹平。Rudin 等人（2021）的 *Learning to Walk in Minutes* 报告：单张 GPU 上，平地行走几分钟、复杂地形约二十分钟量级就能训出来。
- **接口选对**：策略不直接输出扭矩，而是输出关节目标角度，由高频 [PD 控制](#/learn/ctrl/pd) 去跟踪，相当于给策略配了一层稳定的"脊髓反射"。
- **奖励是合同**：正向奖励几乎只有"按指令速度走"，其余全是惩罚，用来排除那些仿真里能拿分、真机上会坏的作弊步态。
- **课程 + 随机化**：地形和速度指令先易后难；摩擦、质量、推力、电机强度全部随机化，逼策略学会不依赖某一组特定的物理参数。

这套配方的参考实现是 legged_gym（Isaac Gym 时代，Isaac Gym 现已停止维护）、它的继任者 Isaac Lab，以及配套的 PPO 实现 rsl_rl。下文可以当作读这些代码时的地图（入门概念见 [运动控制基础](#/learn/loco/basics)）。

## 形式化：把"按指令行走"写成 MDP

策略以速度指令为条件（MDP 概念见 [MDP](#/learn/rl/mdp)）：

$$
a_t \sim \pi_\theta(\cdot \mid o_t), \qquad c_t = (v_x^{*},\, v_y^{*},\, \omega_z^{*})
$$

其中观测 $o_t$ 里包含指令 $c_t$。

动作经过缩放加到默认姿态上，作为 PD 的目标：

$$
q^{\mathrm{des}}_t = q_{\mathrm{default}} + \kappa\, a_t, \qquad \tau = \mathrm{clip}\Big(K_p\big(q^{\mathrm{des}}_t - q\big) - K_d\,\dot q,\; \pm\tau_{\max}\Big)
$$

$\kappa$ 就是 `action_scale`。典型时序是物理步长 $\Delta t_{\mathrm{sim}} = 0.005$ s，`decimation` $=4$，即每个动作保持 4 个物理步、PD 每步重算，策略周期因此是 $4 \times 0.005 = 0.02$ s（50 Hz）。

主奖励是带指数核的速度跟踪项：

$$
r_{\mathrm{lin}} = \exp\!\Big(-\frac{\lVert v^{*}_{xy} - v_{xy}\rVert^2}{\sigma}\Big), \qquad r_{\mathrm{ang}} = \exp\!\Big(-\frac{(\omega^{*}_z - \omega_z)^2}{\sigma}\Big)
$$

总奖励为 $r_t = \sum_k w_k\, r_k\, \Delta t$（legged_gym 把权重乘以策略周期，权重的含义因此与控制频率无关）。折扣 $\gamma = 0.99$ 时有效视界约 $1/(1-\gamma) = 100$ 步，也就是约 2 s。策略只需要"看清"两秒内的后果，这正好是几个步态周期的长度。

## 方法

### 1. 并行仿真 + PPO：一次迭代长什么样

legged_gym / rsl_rl 的默认配置大致是：4096 个环境，每个环境每次迭代只滚 24 步，合计约 10 万条 transition；随后 PPO 做 5 个 epoch × 4 个 mini-batch 的更新，GAE 取 $\gamma=0.99,\ \lambda=0.95$，clip 0.2，学习率按 KL 自适应调整，actor 和 critic 都是 [512, 256, 128] 的 ELU MLP。rollout 很短也没关系：价值函数 bootstrap 补上了后面的回报，而巨大的并行宽度保证了样本多样性。

### 2. 观测设计

以 12 自由度四足为例（legged_gym 默认）：

| 观测项 | 维度 | 真机来源 | 说明 |
| --- | --- | --- | --- |
| 基座线速度 $v_b$ | 3 | 状态估计器 | legged_gym 默认放进 actor（ANYmal 的估计器较可靠）；没有好估计器的平台常只给 critic，或另行估计 |
| 基座角速度 $\omega_b$ | 3 | IMU 陀螺仪 | 机体系，注意 IMU 安装朝向 |
| 投影重力 projected gravity | 3 | IMU 姿态 | $g_b = R^\top(0,0,-1)$，等价于 roll/pitch，没有 yaw 歧义 |
| 速度指令 $c_t$ | 3 | 遥控器 / 上层规划 | $v_x^{*}, v_y^{*}, \omega_z^{*}$ |
| 关节位置 $q - q_{\mathrm{default}}$ | 12 | 编码器 | 相对默认姿态 |
| 关节速度 $\dot q$ | 12 | 编码器差分 | 噪声最大的一项 |
| 上一步动作 $a_{t-1}$ | 12 | 策略自身 | 让策略知道自己"刚才在干嘛"，便于输出平滑 |
| 高度扫描 height scan | 187（可选） | 高程图 | 基座周围网格点的地形高度，依赖感知，见 [感知运动](#/learn/loco/perceptive) |

合计 48 维，带高度扫描时为 235 维。每一项都乘以固定的 `obs_scale` 再 clip，并在训练时加均匀噪声。**这些缩放系数和噪声也是策略的一部分**，部署端必须一模一样。人形常额外加入步态相位时钟 $(\sin\phi, \cos\phi)$，或者堆叠几帧历史观测。

### 3. 动作接口：为什么是位置偏移 + PD

- **高频稳定环**：PD 在仿真里 200 Hz 运行，在真机上跑在驱动器里（kHz 量级），策略只需 50 Hz 给出"意图"。
- **好的探索起点**：$a=0$ 就是默认站姿，初始化的高斯策略一开始就在"站着"附近探索。
- **柔顺**：$K_p$ 通常设得比较低（legged_gym 的 A1 配置是 $K_p=20$ N·m/rad，$K_d=0.5$ 量级），所以 $q^{\mathrm{des}}$ 并不是关节最终到达的位置，而是一根虚拟弹簧的设定点，策略实际上是通过 $q^{\mathrm{des}} - q$ 间接控制扭矩。

代价是 $K_p$、$K_d$、`action_scale` 和 PD 的实现方式都绑进了策略。换增益相当于换了一台机器人。执行器建模细节见 [执行器](#/learn/hw/actuators)。

### 4. 奖励项

| 项 | 形式（示意） | 符号 | 作用 |
| --- | --- | --- | --- |
| 线速度跟踪 | $\exp(-\lVert v^{*}_{xy}-v_{xy}\rVert^2/\sigma)$ | + | 主任务 |
| 角速度跟踪 | $\exp(-(\omega^{*}_z-\omega_z)^2/\sigma)$ | + | 主任务（转向） |
| 竖直速度 | $v_z^2$ | − | 别蹦跳 |
| 横滚/俯仰角速度 | $\omega_x^2+\omega_y^2$ | − | 机身别晃 |
| 姿态 | projected gravity 的 xy 分量平方和 | − | 机身水平 |
| 扭矩 | $\sum \tau^2$ | − | 能耗/发热 |
| 关节加速度 | $\sum \ddot q^2$ | − | 平滑 |
| 动作变化率 | $\sum (a_t - a_{t-1})^2$ | − | 抑制高频抖动，对真机最关键的惩罚之一 |
| 足部腾空时间 | 着地瞬间奖励 $\sum(t_{\mathrm{air}} - 0.5)$，仅在指令非零时生效 | + | 迈大步，不拖脚 |
| 非足部碰撞 | 大腿、小腿触地次数 | − | 别用膝盖走 |
| 关节限位 | 超出软限位的量 | − | 保护关节 |
| 终止 | 非超时的提前终止 | − | 别摔 |

两个容易忽略的细节。第一，指数核把跟踪奖励限制在 $[0,1]$，$\sigma$ 决定"多近才算好"：太小时，离目标远的样本几乎拿不到梯度信号；太大时，跟踪精度又会变差。第二，legged_gym 默认 `only_positive_rewards`，把每步总奖励截断到 $\ge 0$。否则训练初期惩罚占主导、每步奖励为负，策略会发现"早死早超生"。奖励设计的更多套路见 [奖励设计](#/learn/loco/reward)。

### 5. 课程

**地形课程（game-inspired curriculum）**。地形被排成网格：行是难度等级，列是地形类型（平滑斜坡、粗糙斜坡、上楼梯、下楼梯、离散障碍等）。每个机器人分配到一个格子，episode 结束时：走出的距离超过格子边长的一半就升一级；走出的距离不到指令速度对应距离的一半就降一级；已经通关最高级的机器人会被随机分到任意等级，防止遗忘简单地形，也让整个网格始终有机器人分布。整个过程不需要手工调度，难度自动跟着能力走。

**指令课程**。先在小速度范围内采样指令，速度跟踪奖励达到上限的一定比例（legged_gym 用 80%）后再扩大范围。

### 6. 域随机化

| 随机化项 | 典型做法（量级，近似） | 对付的 gap |
| --- | --- | --- |
| 地面摩擦 | 每环境缩放，legged_gym 默认约 0.5–1.25 | 地面材质 |
| 基座附加质量 | ±1 kg 量级，也可随机化质心 | 载荷、建模误差 |
| 推扰 push | 每隔十几秒随机给基座一个速度（~1 m/s 量级） | 外部扰动、恢复能力 |
| 电机强度 | 扭矩缩放 ±10–20% | 电机个体差异、电压、温度 |
| $K_p$ / $K_d$ | 小幅缩放 | 驱动器实现差异 |
| 控制延迟 | 动作延迟 0 到若干个物理步 | 通信与计算延迟 |
| 观测噪声 | 每项按 scale 加均匀噪声 | 传感器噪声 |
| 初始状态 | 关节角、基座速度随机 | 状态覆盖 |

Isaac Lab 把这些写成 `EventTerm`，按 `startup`（启动时一次）、`reset`（每次重置）、`interval`（定时触发）三种模式执行。随机化范围不是越宽越好，太宽会让策略变得保守、步态难看。原则见 [域随机化](#/learn/loco/dr) 和 [sim-to-real](#/learn/sim/sim2real)。

### 7. 终止、超时与重置

- **终止（termination）**：基座触地；人形还常加骨盆高度过低、倾角过大。这是真正的失败，其后的价值为 0。
- **超时（time-out）**：legged_gym 默认 episode 为 20 s。超时**不是失败**，只是人为截断，要用 $V(s_{T})$ bootstrap，rsl_rl 里的写法是 `r += γ · V(s) · time_out`。如果当成终止，价值函数会在 episode 末尾系统性低估，策略也会学到奇怪的"末期行为"。
- **重置**：重新随机化初始关节角、基座速度和地形等级；指令每隔约 10 s 在 episode 内部重新采样一次，所以一个 episode 里会经历多个指令。

### 8. 特权信息与非对称 actor-critic

critic 只在训练时使用，因此可以看真机拿不到的东西：真实线速度、摩擦系数、附加质量、接触力、无噪声的高度图。这就是**非对称 actor-critic（asymmetric actor-critic）**，rsl_rl 原生支持给 critic 单独一组观测。更进一步的做法是**teacher–student**：先训练一个能看特权信息的 teacher，再让只看本体感知历史的 student 去模仿它（Lee et al. 2020），或者用 RMA（Kumar et al. 2021）从历史中在线估计环境隐变量。详见 [teacher–student](#/learn/loco/teacher) 和 [RMA](#/learn/loco/rma)。

### 9. 一次 `env.step` 的骨架

```python
def step(self, actions):                      # 策略 50 Hz 调用一次
    self.actions = actions.clip(-clip_a, clip_a)
    for _ in range(self.decimation):          # 4 × 0.005 s 物理步
        q_des = self.default_q + self.action_scale * self.actions   # (可在此插入延迟缓冲)
        tau = self.kp * (q_des - self.q) - self.kd * self.qd
        tau = tau.clip(-self.tau_limit, self.tau_limit)           # 扭矩饱和
        self.sim.set_joint_torques(tau)
        self.sim.step()
        self.refresh_state()                  # q, qd, 基座位姿, 接触力
    # ---------- post physics ----------
    self.update_derived_state()               # 机体系速度、projected gravity、足部腾空时间
    self.check_termination()                  # 基座触地 → reset；到 20 s → time_out
    rew = self.compute_reward()               # sum_k w_k * r_k * dt，可截断到 >= 0
    env_ids = self.reset_buf.nonzero()
    self.update_terrain_curriculum(env_ids)
    self.reset_idx(env_ids)                   # 域随机化 + 随机初始状态 + 重采样指令
    self.maybe_resample_commands()            # 约每 10 s
    self.maybe_push_robots()                  # 约每十几 s
    obs = self.compute_obs()                  # 缩放 → 拼接 → clip → 加噪声
    priv_obs = self.compute_privileged_obs()  # 给 critic
    self.last_actions = self.actions
    return obs, priv_obs, rew, self.reset_buf, {"time_outs": self.time_out_buf}
```

训练循环端（rsl_rl）就是：每个环境滚 24 步，把 `time_outs` 的 bootstrap 加进奖励，计算 GAE，然后做 PPO 更新，如此反复。Isaac Lab 把上面这些步骤拆成 Observation / Reward / Event / Curriculum / Termination / Command 等 manager，结构不同，内容就是这张配方。

## 机器人视角

> [!robot] **平台**：legged_gym 原论文在 ANYmal C 上做了真机验证，ANYmal 的 SEA 关节用 actuator net 建模。Isaac Lab 的 `Isaac-Velocity-*` 系列任务提供了 ANYmal、Unitree Go2、H1、G1 等机器人的配置；Unitree 官方的 unitree_rl_gym 基于 legged_gym，覆盖 Go2/H1/G1，并带 MuJoCo sim2sim 和部署示例。
> **人形的常见改动**：加步态相位时钟和与之匹配的接触奖励、抬脚高度和足底放平奖励、髋关节偏航/横滚偏离惩罚，终止条件更严格。人形的稳定裕度远小于四足，域随机化和延迟建模更要认真对待，见 [人形运动](#/learn/loco/humanoid)。

### 部署清单

1. **导出**：只导出 actor 的均值动作（不采样），格式用 TorchScript 或 ONNX（legged_gym 和 Isaac Lab 的 play 脚本都能导出）。如果训练时用了经验归一化（empirical normalization），把 normalizer 一起导出。
2. **观测一致**：顺序、`obs_scale`、clip、单位（rad、rad/s）、坐标系（机体系）都要一致。`last_action` 喂策略的原始输出，开机时置零；指令同样要缩放。
3. **关节映射**：仿真的 DOF 顺序和 SDK 的电机顺序通常不同（连 Isaac Gym 与 Isaac Lab 对同一个 URDF 的关节排序都可能不一样），要按关节名建一张映射表，并逐关节验证方向和零位。
4. **默认姿态与 PD**：`default_q`、`action_scale`、$K_p$/$K_d$（确认是关节侧定义、单位一致）、扭矩上限全部与训练配置相同。
5. **时序**：策略由定时器严格按 50 Hz 触发，不要"算完就发"。实测循环抖动（jitter）和端到端延迟，确认落在训练时随机化的范围内。
6. **安全**：先吊架或挂保护绳；软启动，即用 2–3 s 插值到默认姿态后再切换到策略；姿态倾角、关节限位或扭矩超限时切入阻尼模式（$K_p=0,\ K_d>0$）；加通信 watchdog 和物理急停。厂商自带的高层运动服务要先关掉，再切换到底层控制模式（例如 Go2 需要先关闭 sport mode）。

工程细节见 [部署](#/learn/sim/deploy)。

### Sim-to-real 调试清单

1. **sim2sim**：用**部署代码本身**在另一个仿真器（如 MuJoCo，见 [仿真引擎](#/learn/sim/engines)）里跑策略。在 Isaac 里训练、在 MuJoCo 里能走，说明策略没有过拟合 Isaac 的物理细节，部署代码也没写错。
2. **吊起来逐关节**：给每个关节单独发小幅正弦，检查方向、零位、限位。
3. **静止对比观测**：真机和仿真摆同一个姿态，逐维对比观测向量。projected gravity 应接近 $(0,0,-1)$，角速度接近 0，关节偏移符号要一致。
4. **开环回放**：把仿真里的动作序列在吊起的真机上回放，对比关节轨迹，从差异中定位 PD、延迟、摩擦的问题（见 [系统辨识](#/learn/sim/sysid)）。
5. **落地**：先零指令站立，再逐步加速度指令，全程记录观测、动作、扭矩和温度日志。

| 症状 | 优先怀疑 |
| --- | --- |
| 一上电就乱踢或翻倒 | 关节顺序、方向、零位，观测拼接顺序 |
| 站立时高频抖动 | 延迟未建模、$K_p$/$K_d$ 不一致、关节速度噪声、`action_rate` 惩罚太弱 |
| 零指令也会漂移 | 线速度估计偏差、IMU 安装偏角、指令零点 |
| 走一会儿越来越软、拖腿 | 电机过热降额、电池压降 |
| 仿真能上台阶，真机不行 | 摩擦/地形随机化不足、足端碰撞几何、高度图噪声 |

## 常见坑

> [!pitfall] **观测"差不多对"**：少乘一个 `obs_scale`、`dof_vel` 单位错、`last_action` 喂了缩放后的值、两条腿的顺序对调……策略对这些错误不会"大致容忍"，结果往往是一上电就乱动。办法是写一个函数，同时从仿真状态和真机状态生成观测，静止对比到每一维。

> [!pitfall] **策略依赖了真机拿不准的量**：actor 里放了基座线速度，可真机上的估计器在打滑、落地冲击时误差很大，策略就会跟着漂。没有可靠估计器时，把线速度只给 critic，或训练一个估计器 / 走 teacher–student。足端接触、无噪声高度图同理。

> [!pitfall] **仿真里看不出的奖励作弊**：高频 bang-bang 抖腿、拖脚滑行、跪着走、某条腿一直不着地。仿真的电机不发热、齿轮不磨损，这些步态照样拿高分。训练时要盯视频，同时画出关节扭矩、功率和动作频谱，`action_rate`、腾空时间、碰撞惩罚要配齐。

> [!pitfall] **把超时当终止**：不做 time-out bootstrap，价值函数会在 episode 末尾系统性低估；再加上总奖励为负，策略就会学会尽早"自杀"结束 episode。超时要 bootstrap，跌倒才算真终止，必要时把每步奖励截断到非负。

## 自测

> [!quiz] `dt = 0.005`、`decimation = 4`、`γ = 0.99` 分别意味着什么？如果真机上的策略循环实际只跑到 40 Hz，会发生什么？
> 物理以 200 Hz 积分，PD 每个物理步重算；策略每 4 步出一次动作，即 50 Hz。$\gamma=0.99$ 对应约 100 个策略步的有效视界，约 2 s。如果真机跑到 40 Hz，每个动作被保持 25 ms 而不是 20 ms，相当于引入了训练中没见过的额外延迟和保持时间。轻则抖动、步频改变，重则失稳。同时 `last_action`、关节速度等观测的统计分布也随之改变。修法是保证定时严格，并在训练时随机化延迟。

> [!quiz] 为什么动作设计成"默认姿态上的关节位置偏移 + PD"，而不是直接输出扭矩？$K_p$ 设低或设高分别会怎样？
> 位置 + PD 提供了一个高频稳定的底层环，50 Hz 的策略就够用；$a=0$ 对应站姿，探索起点好；输出也更平滑，对仿真误差更鲁棒。$K_p$ 低时关节柔顺、抗冲击、对模型误差宽容，但跟踪差，策略必须学会利用 $q^{\mathrm{des}} - q$ 来"间接给扭矩"；$K_p$ 高时跟踪硬，但对延迟和噪声敏感，容易抖动，落地冲击大。无论哪种，增益都成了策略的一部分，真机必须与训练时一致。

> [!quiz] 写出 time-out 的正确处理方式，并解释为什么它和终止不同。
> 设最后一步的奖励为 $r_T$。若是终止（跌倒），回报目标是 $r_T$，后续价值为 0。若是超时，这个状态本身并没有结束，未来本可以继续拿奖励，应该用 $r_T + \gamma V(s_{T+1})$ 作为目标（rsl_rl 在奖励里加 $\gamma V(s)\cdot \mathbb{1}[\text{timeout}]$）。如果混为一谈，价值函数就得根据一个观测里看不到的计时器预测"什么时候会突然归零"，值估计偏低且噪声大，进而污染 advantage。

> [!quiz] 描述 legged_gym 地形课程的升降级规则。为什么通关最高级的机器人要被随机分配到任意等级？
> 每个 episode 结束时：走出的距离超过格子边长的一半就升级；不到指令速度对应距离的一半就降级。通关最高级的机器人会被随机分到任意等级，原因有两个：一是防止所有机器人挤在最难的等级、忘掉简单地形（灾难性遗忘）；二是让各个难度上始终有样本，维持训练数据的多样性。这个机制让难度自动匹配当前能力，无需手调课程表。

> [!quiz] 策略在仿真里完美，真机一站起来就剧烈高频抖动。按什么顺序排查？
> 先排除"接错线"类问题：关节顺序和方向、观测缩放和顺序（静止对比观测向量）、`default_q` 与 $K_p$/$K_d$ 是否一致。然后查时序：实测循环频率和延迟，是否超出训练时的延迟随机化范围。再查信号：关节速度观测是否噪声过大、是否和仿真同样做了滤波或差分。最后才回到训练端：延迟随机化、`action_rate` / 关节加速度惩罚、降低 $K_p$，或者加大观测噪声重新训练。顺序的原则是先查便宜且确定的，后查昂贵且模糊的。

## 延伸阅读

- [Rudin et al. 2021, Learning to Walk in Minutes Using Massively Parallel Deep RL](https://arxiv.org/abs/2109.11978)：这套配方的出处，讲清了大规模并行下 PPO 的超参选择和游戏式地形课程。
- [legged_gym](https://github.com/leggedrobotics/legged_gym)：配方的参考实现，`legged_robot.py` 和 `legged_robot_config.py` 两个文件值得逐行读。
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)：配套的精简 PPO 实现，代码短，适合对照公式读。
- [Isaac Lab 文档](https://isaac-sim.github.io/IsaacLab)：新一代框架，manager-based 环境把这套配方拆成可组合的模块，locomotion velocity 任务是最好的入口。
- [Hwangbo et al. 2019](https://arxiv.org/abs/1901.08652)：actuator net 与 ANYmal 上的早期 sim-to-real 工作，是理解"为什么要建模执行器"的经典。
- [Kumar et al. 2021, RMA](https://arxiv.org/abs/2107.04034)：不靠特权信息部署，而是在线估计环境隐变量的另一条路线。

> [!mine] 我的笔记：学完后在这里写下自己的理解、疑问和实验记录。
