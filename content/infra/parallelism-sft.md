## 推理和训练是两套并行

实习时最容易混的一点：vLLM 里的并行和训练里的并行**不是一回事**，两套系统只通过 JSON 数据和 LoRA 权重文件连在一起。

### 推理（vLLM）

| 并行方式 | 切什么 | 代价 |
| --- | --- | --- |
| TP（张量并行） | 同一层里的矩阵切到多卡 | 每层都要 all-reduce，适合有 NVLink 的单机 |
| PP（流水线并行） | 按层切 | 有流水线气泡 |
| DP（数据并行） | 整个引擎复制多份 | 只为提高吞吐，**没有梯度同步** |
| EP（专家并行） | MoE 的专家分到不同卡 | 只对 MoE 有用 |

「MP（模型并行）」是 TP 和 PP 的统称，不是一个单独的设置。

7B 模型单机推理：TP = 1–2 最划算；没有 NVLink 时，PP 可能比 TP 快。

### 训练

- **DDP**：每张卡一份完整模型，反向传播后 all-reduce 梯度；
- **FSDP / DeepSpeed ZeRO**：把参数、梯度、优化器状态切到多卡，放得下更大的模型；
- 启动用 `accelerate launch` 或 `torchrun`。

> [!pitfall] `device_map="auto"` 不是分布式训练。它只是把一个模型的层摊到几张卡上，一次只有一张卡在算。要多卡训练，就去掉它，用 `accelerate launch` / `torchrun` 启动。

## LoRA

不改原始权重 $W$，只学一个低秩增量：

$$
W' = W + \frac{\alpha}{r} BA, \qquad B \in \mathbb R^{d \times r},\ A \in \mathbb R^{r \times k},\ r \ll \min(d, k)
$$

$B$ 初始化为 0，所以训练开始时模型和原来完全一样。可训练参数从 $dk$ 降到 $r(d+k)$。

## 实习时的 SFT 配方

- **只在回答上算 loss**：prompt 部分的 label 设成 $-100$（交叉熵会忽略这个值），只学目标回答和 EOS；
- LoRA：$r = 16$，$\alpha = 32$，挂在所有线性层（`all-linear`）；
- bf16；
- 训练 / 验证 99 / 1 划分，固定随机种子 42；
- 每个 epoch 只存 adapter，不存整个模型。

## 工具的取舍

LLaMA-Factory 把 HuggingFace 那一套封装好了，上手快；但要做细的消融（改 loss、改数据流），直接写 HF + accelerate 更透明。
