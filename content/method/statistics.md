## 数据怎么划分

TotalSegmentator v2.0.1 一共 1,228 个 CT，672 个通过质量控制，其中 417 个从背后能看到完整的 L1–L5。**按病人**划分：

| | 病人数 | 用来做什么 |
| --- | --- | --- |
| 训练 | 292 | 4 个世界模型预测器 |
| 其中 | 252 | 人群 L4 目标、运动映射 $A$ |
| 再其中 | 227 | 度量头和检索库（90,800 帧） |
| （训练里留出） | 40 | 开发调试 |
| 验证 | 42 | 所有模型和配置的选择 |
| 测试 | 83 | 只跑一次 |

同一个病人的帧绝不会同时出现在训练和测试里。

## 先声明，再测试

- 所有选择都在验证集上做：世界模型选哪个组合、检索用单帧还是三帧；
- 在任何检索控制器碰测试集**之前**，先写下要做哪些比较。主比较是：单帧检索 vs. 选出的世界模型，看 Dwell30；
- 事后才加的比较，报告时标明「事后、未校正」。

这样测试集上的 p 值才有意义，否则就是在测试集上挑结果。

## 配对 + 按病人 bootstrap

同一个病人的 3 个 episode 之间是相关的（同一副骨架、同一种体型），**病人才是独立的单位**。所以：

1. 每个病人内部先汇总（距离取中位数，二值结果取比例）；
2. 两个策略在同一个病人上的结果做差，得到配对差；
3. 有放回地重抽**病人**（不是 episode），抽 2,000 次，每次算配对差的均值，取 2.5% 和 97.5% 分位数作为 95% CI。

按 episode 重抽会低估方差，CI 会窄得不真实。

```python
import numpy as np

def paired_patient_bootstrap(a, b, n_boot=2000, seed=0):
    """a, b: 每个病人的汇总值（同一顺序）。返回均值差和 95% CI。"""
    rng = np.random.default_rng(seed)
    d = np.asarray(b, float) - np.asarray(a, float)   # 配对差，每个病人一个
    n = len(d)
    boots = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return d.mean(), np.percentile(boots, [2.5, 97.5])
```

## 多重比较：Holm

同时做 $m$ 个检验，每个都用 $\alpha = 0.05$，至少错一个的概率（family-wise error rate）会远高于 5%。

**Holm 步降法**：

1. 把 $p$ 值从小到大排：$p_{(1)} \le \dots \le p_{(m)}$；
2. 第 $i$ 个和 $\alpha / (m - i + 1)$ 比；
3. 从最小的开始，能拒绝就继续，**第一次拒绝不了就停**，后面的都不拒绝。

调整后的 $p$ 值：$\tilde p_{(i)} = \max_{j \le i} \min\big(1, (m - j + 1)\, p_{(j)}\big)$。它和 Bonferroni 一样控制 FWER，但永远不会更保守。

```python
def holm(pvals):
    p = np.asarray(pvals, float); m = len(p)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj
```

## 这个项目里的真实例子

- 主比较 Dwell30：检索高 7.2 个点，95% CI [0.8, 14.1]，**不含 0**。但在 3 个预先声明的检验里做 Holm 校正后，$p = 0.068$，**不显著**。论文照实写「did not survive Holm correction」；
- 消融的 Dwell30：打乱运动映射调整后 $p = 0.015$（显著），打乱检索库 0.27，去掉度量头 0.69；
- 终点类的差别（Final-5 少 11.8 mm、L4 final 多 24.9 个点）CI 都离 0 很远，但它们是事后比较，标为未校正。

> [!note] CI 不含 0 和校正后显著是两回事。前者只看这一个比较，后者考虑了你一共做了几个比较。报告时两个都给，并说清楚哪个是预先声明的。
