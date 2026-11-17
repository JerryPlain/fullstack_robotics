## 为什么用 vLLM

评测和数据生成要对几万到几十万条 prompt 跑模型。逐条调用 HuggingFace 的 `generate` 太慢；vLLM 的离线批量推理快得多，靠两样东西：

- **PagedAttention**：把 KV cache 切成固定大小的块来管理，像操作系统的分页内存，显存碎片少，能同时塞更多请求；
- **continuous batching**：一个请求生成完就立刻补进新的，不用等整批里最长的那条。输出长度差别很大的评测任务特别受益。

## 显存都花在哪

- **prefill**：把整个 prompt 过一遍，建好 KV cache；
- **decode**：一个 token 一个 token 地生成，每步读 KV cache；
- OOM 很少是因为参数本身，通常是**权重 + KV cache + 并发数 + 最大输出长度**加在一起超了。

## 实习时摸出来的稳定配置

| 设置 | 为什么 |
| --- | --- |
| `tensor_parallel_size = GPU 数`，PP = 1 | 单机推理用张量并行 |
| `max_model_len=8192` | 按任务实际需要设上限，别按模型支持的最大长度预留 KV cache |
| `gpu_memory_utilization=0.9` | 0.85–0.92 之间比较稳 |
| `enable_chunked_prefill=True` | 长 prompt 分块 prefill，削掉显存尖峰 |
| `enforce_eager=True` | 不做 CUDA graph 捕获，配 LoRA、长上下文时更稳 |
| `disable_custom_all_reduce=True` | 回退到普通 NCCL，少出怪错 |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | 避免 CUDA 和 fork 冲突 |
| 运行时用 `LoRARequest` 挂 adapter | 不用先合并权重 |

还可以用 `max_num_batched_tokens`、`max_num_seqs` 限制调度器一次处理的量。

**OOM 时按这个顺序调**：降 `max_model_len` → 降显存占用比例 → 缩短 prompt 或 `max_tokens` → 去掉 LoRA → 用 TP=1 单卡试。

启动前检查：`num_attention_heads % tensor_parallel_size == 0`，不整除就起不来。

## 当裁判时的输入截断

judge 模型的上下文有限，prompt + response 太长时要截断。实习时 Qwen3PrivacyGuard 的做法：

- 先量出模板本身占多少 token，再留 20 个 token 的余量；
- 问题最多占预算的 25%；
- **保留问题的结尾、回答的开头**：问题的关键要求通常在最后，回答是否顺从通常在开头就看得出来；
- 直接传 `prompt_token_ids`，避免二次分词带来的长度误差。

## 结构化输出

让 judge 用固定格式输出（XML 或 JSON），解析失败统一记成 `{}`。这样失败率可以单独统计，不会悄悄混进结果里。
