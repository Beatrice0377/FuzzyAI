[English](README.md) | **简体中文**

# FuzzyAI

**状态：早期开发阶段（Status: early development）。** 确定性内核（deterministic core）与 Bool 垂直切片（vertical slice，真实本地 Hugging Face 后端）均已实现。Choice 推理、校准（calibration）与弃权（abstention）尚未实现。

FuzzyAI 是一个 provider-agnostic（供应商无关）的概率决策运行时（probabilistic decision runtime）。它把语言模型变成可评估（evaluable）、可校准（calibratable）、可追踪（trackable）的语义概率决策组件。

要解决的问题：LLM 被接入程序逻辑时，人们常把一坨原始文本或一个未经审视的分数当作可信的概率。结果是：系统说不清一个分数意味着什么，无法复现过去的决策，也无法区分"模型不确定"与"程序应当拒绝行动"。FuzzyAI 为这一领域提供一个狭窄而确定性的内核。

它不是聊天框架，不是 agent 框架，也不仅仅是 structured-output（结构化输出）的包装层。

## 核心原则

> **LLM 处理语义不确定性。程序处理确定性策略。**
> The LLM handles semantic uncertainty. The program handles deterministic policy.

## 当前可用：Bool 垂直切片

目前唯一端到端可用的路径是 Bool-only。一个轻量的 `FuzzyAI` facade 按固定顺序编排整条流水线，自身不添加任何回退（fallback）：

```
BoolDecision -> BoolCompiler -> InferencePlan -> TransformersBackend
   -> RawEvidence -> assemble_bool_probability -> BoolResult -> DecisionTrace
```

后端是可选附加包（optional extra）。基础包没有任何运行时依赖；只有安装附加包才会引入 `torch` 与 `transformers`：

```bash
uv sync --extra transformers    # 依赖组：torch>=2.7, transformers>=4.53
```

```python
from fuzzyai import BoolDecision, FuzzyAI
from fuzzyai.backends.transformers import TransformersBackend

backend = TransformersBackend(
    "Qwen/Qwen3-0.6B",  # 一个本地 HF causal LM
    # 否则 Qwen3 会先输出推理块；见下方说明。
    chat_template_kwargs={"enable_thinking": False},
)
ai = FuzzyAI(backend=backend)

result = ai.evaluate(
    BoolDecision(
        question="Does the evidence support the claim?",
        context={"claim": "the cache was warm", "evidence": "hit ratio rose"},
    )
)

result.probability_true        # 两个 verbalizer logit 上的 two-way softmax
result.certainty               # 该分布的熵 entropy + margin
result.predicted_correctness   # None：校准尚不存在
result.trace_id                # 将结果与其 DecisionTrace 关联起来
```

`ai.evaluate_with_trace(decision)` 返回一个 `Evaluation`（结果加上它的 `DecisionTrace`）。trace 记录了 decision 与 plan 的指纹（fingerprint）、scoring strategy、doctrine id、解析出的 verbalizer token id、原始证据（raw evidence）、input fingerprint、backend 类型与延迟（latency）。

`probability_true` 是什么：在恰好两个 verbalizer-token logit 上做的数值稳定的 two-way softmax，等于 `sigmoid(l_true - l_false)`。它不是什么：不是全词表（full-vocabulary）概率，不是现实世界事件概率，不是预测准确率，也不是校准后的数值。`calibrated` 恒为 `False`，`predicted_correctness` 恒为 `None`。

后端加载一个本地 causal LM，在 `eval()` 与 `torch.inference_mode()` 下运行，优先使用 CUDA 并回退到 CPU，且只读取最后一个位置的 logits。它从不调用 `generate()`，从不解析生成的文本。verbalizer 默认为 `yes`/`no`，且各自必须在真实的 chat-template 前缀之后解析为恰好一个、互不相同的打分 token；否则抛出 `VerbalizerError`。没有静默回退，没有截断，没有多 token logit 求和，也没有采样回退。

`probability_true` 只有在模型确实处在决策位置时才有意义，也就是说它的下一个 token 真的必须是两个 verbalizer 之一。某些模型（thinking / reasoning 模型）会先输出推理块：如果 chat-template 的渲染模式不对，模型概率最高的 token 会是它的推理开标签，两个 verbalizer token 都落在分布极尾部，两路 softmax 于是把尾部噪声重归一化成一个看起来合理的数字。上面的示例因此使用 `chat_template_kwargs={"enable_thinking": False}` 渲染。实测差异记录在 [docs/claims.md](docs/claims.md)：同一个 plan、同一个模型，thinking 模式下 `P(True) = 0.3479`，关闭 thinking 后 `P(True) = 0.9951`。Phase 2A 不会自动检测这种情况，而「拒绝作答」属于尚不存在的 policy 行为。

decision 的 context 只会被渲染进 user prompt 作为证据（evidence），永远不会进入 system prompt。这是一条结构性放置规则，不是 prompt-injection（提示注入）安全性声明。

`ChoiceDecision` 没有运行时路径：它仍是 Phase 1 的数据模型，编译它会抛出 `UnsupportedDecisionError`。

## Planned API（尚未实现）

下面的便捷接口仍是 **planned API（尚未实现）**。不存在 `ai.bool` 或 `ai.choice` 入口；当前可用的调用是上文所示的 `ai.evaluate(BoolDecision(...))`。

```python
risk = ai.bool("Is this transaction suspicious?", context=transaction)

route = ai.choice(
    "Which team should handle this ticket?",
    choices={
        "billing": "Payment and billing issues",
        "shipping": "Delivery and logistics",
        "returns": "Returns and refunds",
    },
    context=ticket,
)
```

## 概率、certainty、predicted correctness：不是一回事

- **概率（Probability）**：由某种 scoring strategy（打分策略）产生的决策分布。它并不自动等于现实世界中的正确率。当前 Bool 概率来自 binary token logits，不是全词表分布。
- **Certainty**：只描述该概率分布有多集中（熵 entropy、margin）。它是一个数学属性，不是正确性概率。我们从不把它称为 "confidence"。规范性约束：certainty 不得被称为 confidence（置信度）。
- **Predicted correctness（预测正确性）**：只有在有效校准之后才可表达。当前产出的每个结果都是 `predicted_correctness = None` 且 `calibrated = False`。max softmax、logit、熵，或 LLM 自称"我有 95% 把握"，都不是 predicted correctness。

完整的规范性定义见[设计宪法](docs/design-constitution.md)。

## 当前已实现的内容

Phase 1 确定性内核：

- `BoolDecision` 与 `ChoiceDecision` 决策规格
- 结果模型：`BoolResult`、`ChoiceResult`、`Certainty`
- `BackendCapabilities`（显式数据）与狭窄的 `Backend` Protocol
- `InferencePlan` 与 `RawEvidence` 抽象
- 确定性指纹系统（canonical JSON + SHA-256）
- 以 `FuzzyAIError` 为根的错误分类体系

Phase 2A Bool 路径：

- `ScoringDoctrine` 与默认 doctrine `BINARY_SEMANTIC_JUDGMENT_V1`
- `BoolCompiler`（纯规划器：它从不计算概率）
- `assemble_bool_probability`（binary token logits 上的 two-way softmax）
- `TransformersBackend`（可选 `transformers` 附加包，只读 logits）
- `DecisionTrace` / `build_decision_trace`（其 `trace_id` 从不派生自任何指纹）
- 轻量的 `FuzzyAI` / `Evaluation` facade

```
BoolDecision --(BoolCompiler)--> InferencePlan --(TransformersBackend)--> RawEvidence
                                                                             |
                              assemble_bool_probability     (calibration: still future)
                                                                             v
                                                BoolResult + DecisionTrace (via FuzzyAI)
```

Choice 推理、校准与弃权仍是未来工作。

## 证据与主张（Evidence and claims）

性能（performance）、质量、校准（calibration）与 provider 支持相关的声明，统一按证据状态（evidence status）记录在 [docs/claims.md](docs/claims.md) 中。本项目目前不做任何 benchmark、性能或模型支持声明：Bool 路径可以对着本地 Hugging Face causal LM 运行，但没有任何模型被评估过。假设（hypotheses）与路线图条目在登记册中均被如实标注，不作为已实现的能力呈现。

## 开发

本项目自身的开发命令（需要 [uv](https://docs.astral.sh/uv/)）：

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

若未运行 `uv sync --extra transformers`，`transformers` 后端测试会自动跳过；这些测试从不下载模型，也不需要 GPU。

## 文档

- [docs/design-constitution.md](docs/design-constitution.md)：具有约束力的设计约束、术语表与规范性不变量。提出任何方案之前请先阅读。
- [docs/roadmap.md](docs/roadmap.md)：预期的阶段划分。不是时间表，也不是承诺。
- [docs/claims.md](docs/claims.md)：每一项能力主张的证据状态（[V] 已验证、[E] 实验性、[H] 假设、[R] 路线图）。

## 非目标（Non-goals）

仍然没有 HTTP，没有 OpenAI/Anthropic/vLLM/SGLang 集成，没有云端后端，没有自动路由，没有 decision graph，没有校准算法，没有弃权策略，也没有 dashboard、server、agent、RAG、数据库、telemetry 或 web UI。本项目不做任何 benchmark 或性能声明，也不声称任何模型质量或模型支持。

## 许可证

MIT。见 [LICENSE](LICENSE)。
