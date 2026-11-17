## CEM 是什么

交叉熵方法（cross-entropy method, CEM）是一种**不用梯度**的随机优化方法。维护一个动作分布（通常是对角高斯），反复做四件事：

1. 从 $\mathcal N(\mu, \sigma^2)$ 里采 $N$ 个候选动作；
2. 给每个候选算代价；
3. 留下代价最低的 $K$ 个，叫**精英**（elites）；
4. 用精英的均值和标准差更新 $\mu$、$\sigma$。

迭代 $I$ 轮后，分布收缩到低代价的区域，返回 $\mu$。

```python
import numpy as np

def cem(cost, dim=3, n=300, iters=10, n_elite=30, lo=-1.0, hi=1.0,
        init_std=0.5, min_std=0.05, rng=np.random.default_rng()):
    mu, std = np.zeros(dim), np.full(dim, init_std)
    for _ in range(iters):
        a = np.clip(mu + std * rng.standard_normal((n, dim)), lo, hi)  # 采样
        c = np.array([cost(x) for x in a])                              # 打分（实际里要批量算）
        elite = a[np.argsort(c)[:n_elite]]                              # 精英
        mu, std = elite.mean(0), np.maximum(elite.std(0), min_std)      # 重拟合
    return mu
```

`init_std`、`min_std` 是示意值。`min_std` 防止分布过早塌缩成一个点。

## 和 MPC 一起用

CEM 只负责解一次优化，闭环靠**模型预测控制**（MPC，也叫滚动时域控制）：

```python
while not done:
    a = cem(lambda a: cost(a, obs))   # 用当前观测规划
    obs = env.step(a)                 # 只执行一个动作，然后重新规划
```

每一步都用最新的观测重新规划，模型误差和意外扰动就不会一直累积。

## 在这个项目里

- 两个控制器用的是**完全一样**的 CEM：300 个样本、10 轮、30 个精英，每执行一个动作就重新规划；
- **单步**规划：只优化下一步的动作，不优化一整段序列；
- 两者只差 `cost`：世界模型用[特征空间里的外观距离](#/n/wm/latent-planning)，检索用[解剖坐标里的期望距离](#/n/ctrl/belief-planning)。

## 为什么单步贪心有时够用、有时不够

单步规划等于每步都在代价函数上往下走一小步。它够用的前提是：**代价本身就是「离目标还有多远」**，像一个好的价值函数。

- 检索的代价是到 L4 的几何距离（mm）。离目标越近代价越低，单调又平滑，贪心就够了；
- 世界模型的代价是到「L4 长相」的特征距离。L3、L5 也像 L4，代价曲面上有多个低谷，贪心会被困住，或者在附近漂。

所以「规划器不变、只换代价」这个对比设计，本身就说明了问题出在代价上，不在优化器上。

## 要记得的工程细节

- 精英比例一般取 10% 左右（这里 30/300）；
- 动作要裁剪到合法范围；
- 规划多步序列时，用上一步的解平移一格来**热启动**；
- **MPPI** 是它的近亲：不硬选精英，而是按 $\exp(-\text{cost}/\lambda)$ 给所有样本加权。
