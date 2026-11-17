## 开服务器，再跑实验

> serve = 开服务器；run = 调服务器跑实验。

- 一个长时间运行的 SLURM 作业在 GPU 节点上起 vLLM 的 OpenAI 兼容服务（`vllm.entrypoints.openai.api_server`）；
- 评测作业只是**客户端**：设置 `START_SERVER=0` 和 `BASE_URL=http://<节点>:<端口>/v1`，去调这个服务；
- 交互式调试时，用 tmux 挂着服务。

好处：模型只加载一次，可以跑很多组实验；所有开源模型都走同一套 OpenAI 客户端代码。vLLM 本身的原理见 [vLLM 批量推理](#/n/infra/vllm-inference)。

## 要几张卡

| 模型大小 | GPU 数 |
| --- | --- |
| 7–12B | 1 |
| Qwen3-VL-32B、InternVL2.5-26B | ≥ 2 |
| 72B / 78B、Gemma / MedGemma-27B | 4 |

`tensor_parallel_size` 和 GPU 数保持一致。

## 显存：给 CUDA graph 留余量

vLLM 先加载权重，再按 `gpu_memory_utilization` 从**剩下的**显存里预留 KV cache。

设成 0.98 时，只剩约 0.6 GiB 给 CUDA graph 捕获，而它要给 67 种批大小各准备缓冲区，于是 OOM。两种解法：

- 降到 0.90：KV cache 容量从约 208k token 降到 181k；
- 或者 `--enforce-eager`，不捕获 CUDA graph：吞吐低大约 10–20%。

## 和别的模型共用一张卡

智能体节点上还要跑文本编码器、重排序器和 SAM 2，所以 vLLM 的显存比例压到 **0.55**。SAM 2 和 vLLM 抢同一张卡导致 OOM，最后的办法是把 SAM 2 放到另一张卡上。

## 踩过的坑

- **prompt 超长**：「74318 > 65536」。多模态模型的上下文主要被**图像 token** 占掉，图越多、长宽比越极端，token 越多，要限制每次请求的帧数，见 [MLLM 怎么看图](#/n/llm/mllm-basics)；
- **主机名解析失败**：worker 节点的主机名有时解析不了，直接用 IP；
- **客户端作业也申请了 GPU**：结果把双卡节点占住，服务起不来。客户端只需要 CPU；
- `port`、`worker`、`nodelist` 是三个不同的东西，别混。

## 集群上的其他习惯

- 大文件放在专门的工作空间（workspace），不放 home；
- 用 Apptainer 容器跑，`--nv` 挂 GPU，`--bind` 挂数据目录；
- 整个项目大约用了 5,470 GPU 小时（A100 约 4,766，A40 约 703）。算力也是要记账的。
