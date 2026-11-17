## 是什么

粒子群优化（Particle Swarm Optimization, Kennedy & Eberhart 1995）是一种**不用梯度、基于种群**的优化方法。灵感来自鸟群觅食：每只鸟记得自己找到过的最好位置，也知道整群找到过的最好位置。

维护 $N$ 个粒子，每个粒子有：

- 位置 $x_i$：一个候选解；
- 速度 $v_i$：下一步往哪走；
- 适应度 $J(x_i)$：这个解有多好；
- 个人最优 $p_i^*$：它自己见过的最好位置。

整群的最优 $g$ 是所有 $p_i^*$ 里最好的那个。

## 更新公式

$$
v_i^{(t+1)} = \underbrace{\omega\, v_i^{(t)}}_{\text{惯性}} + \underbrace{c_1 r_1 \big(p_i^* - x_i^{(t)}\big)}_{\text{向自己学}} + \underbrace{c_2 r_2 \big(g - x_i^{(t)}\big)}_{\text{向群体学}}
$$

$$
x_i^{(t+1)} = x_i^{(t)} + v_i^{(t+1)}
$$

- $\omega$：惯性权重，平衡探索和利用；
- $c_1, c_2$：认知系数和社会系数；
- $r_1, r_2 \sim U(0, 1)$：随机因子，防止过早收敛。

每轮评估完：$J(x_i)$ 比 $J(p_i^*)$ 好就更新 $p_i^*$，再从所有 $p_i^*$ 里更新 $g$。到固定轮数停下，返回 $g$。

```python
import numpy as np

def pso(J, dim, n=20, iters=100, w=0.729, c1=1.494, c2=1.494, lo=-5, hi=5, rng=np.random.default_rng(0)):
    x = rng.uniform(lo, hi, (n, dim)); v = np.zeros((n, dim))
    p, pJ = x.copy(), np.array([J(xi) for xi in x])        # 个人最优
    g = p[pJ.argmin()].copy()                              # 全局最优（最小化）
    for _ in range(iters):
        r1, r2 = rng.random((n, dim)), rng.random((n, dim))
        v = w * v + c1 * r1 * (p - x) + c2 * r2 * (g - x)
        x = np.clip(x + v, lo, hi)
        f = np.array([J(xi) for xi in x])
        better = f < pJ
        p[better], pJ[better] = x[better], f[better]
        g = p[pJ.argmin()].copy()
    return g
```

## 教科书里的经验值

- 惯性权重是 Shi & Eberhart（1998）加进去的；常用的收缩系数取值 $\omega \approx 0.729$，$c_1 = c_2 \approx 1.494$；
- 速度通常要裁剪，不然会飞出去；
- **gbest 拓扑**（大家都看全局最优）收敛快，但容易早熟；**lbest 拓扑**（只看邻居，比如环形）慢一些，但更多样；
- 最常见的失败：所有粒子都去抄 $g$，多样性塌缩。

## 和其他方法比

| 方法 | 特点 | 什么时候用 |
| --- | --- | --- |
| PSO | 粒子一直存在、移动；只通过 $g$ 共享信息 | 评估便宜、能并行 |
| 遗传算法 | 有选择、交叉、变异；个体会被淘汰 | 解能自然地拼接 |
| 模拟退火 | 单点搜索 | 没法并行探索，也没有群体共享的知识 |
| CMA-ES | 自适应协方差的高斯采样 | 连续、中等维度 |
| 贝叶斯优化 | 用 GP / TPE 建代理模型，选最值得评估的点 | **评估极贵**（只能评估十几次） |

种群方法每轮要评估 $N$ 次，总代价约 $N \times T$ 次评估。在 ReGoRL 的计划里，一次评估要训练几个小时、总共只能跑约 12 次，这时 PSO 和 CMA-ES 大概率吃亏，贝叶斯优化更合适。

## 为什么 SwarmAgentic 选了 PSO

- 搜索空间是**离散的、结构化的、不可微的**（一整套多智能体系统的设计），只能用无梯度方法；
- 评估比较贵，一个有方向引导的种群比盲目变异效率高；
- **选算法的标准**：看它的原语能不能用 LLM 实现，再看它的搜索结构是否匹配问题的成本和地形。PSO 只需要 diff 和 apply 两个原语，见 [语言空间里的优化](#/n/llm/language-space-optimization)。
