## 为什么要协议

智能体每一步的输出要被代码执行：调哪个工具、参数是什么。模型随便输出一段话，环境就没法跑。所以要一份**结构化输出协议**。

每一步的动作统一成：

```json
{"action": "<tool_name>", "params": {...}}
```

不用工具、直接作答时：

```json
{"answer": <按任务类型的答案>}
```

答案的类型跟着任务走：标签列表、三元组字符串、归一化到 [0, 1] 的 (x, y, w, h) 框（或者模型原生的 `bbox_2d`）、评分字典比如 `{"c1": 2, "c2": 1, "c3": 2}`。

## 调用方式 vs 约束方式

一个关键的拆分：**怎么调用模型**（载体）和**怎么约束、解析它的输出**（协议）分开。

每个后端一份 `StructuredOutputContract`，打包四样东西：

1. 追加到 prompt 后面的说明；
2. 请求里的额外参数；
3. 解析器；
4. 解析失败时的修复 prompt。

智能体基类只负责转发这些，不关心具体是哪家的格式。

## 三家后端怎么做

| 后端 | 方法 | 约束在哪一层 |
| --- | --- | --- |
| OpenAI | `response_format` 的 json_schema，`strict: False` | API 层 |
| vLLM | `extra_body` 里的 `guided_json` | **解码层**：逐 token 约束 |
| Claude | 原生 tools + input_schema，定义一个虚拟工具 `emit_agent_step` | API 层 |

- OpenAI 用 `strict: False`，因为不同工具的 `params` 不一样。只严格约束外层的动作结构，参数交给环境里每个工具自己的 schema 校验；
- vLLM 提供 OpenAI 兼容的接口，所以 vLLM 智能体是 OpenAI 智能体的子类；Anthropic 的接口不同（system 放顶层、内容块、计费字段），所以 Claude 智能体直接继承基类。

## 效果

用了解码层约束的开源模型：截断和参数不匹配都是 **0%**。API 模型第一次跑时：GPT-4o-mini 截断 9.7%，Claude 参数不匹配 4.3%，重跑之后降到 0.0% 和 0.1%。

## 踩过的坑

- **guided JSON 和思考模型打架**：约束解码会压制模型原生的思考过程，模型会卡在空白字符上，把 JSON 截断；只能用 frequency_penalty 和长度上限打补丁；
- **重构共享代码**：改了公共的直接作答协议，子类还在调用已经删掉的辅助函数，vLLM 的无工具实验直接报错。改基类之前先搜一遍调用方。

## 论文之后才明白的

论文时期的协议其实是在**模拟**工具调用：把工具说明贴进 prompt，让模型在正文里输出 JSON，下一轮用「Last tool output:」把结果喂回去。

**原生的 tool calling** 是另一套：请求里传 `tools=`，模型返回 `tool_calls[]`，结果以 `role: "tool"` 的消息、带上对应的 `tool_call_id` 返回，必要时 `parallel_tool_calls: false`。

> guided JSON 更适合约束**最终的结构化结果**，不一定适合替代完整的 agent / tool-calling 协议。
