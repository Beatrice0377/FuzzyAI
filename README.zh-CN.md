[English](README.md) | **简体中文**

# FuzzyAI

**状态：早期开发阶段（Status: early development）。** 仅处于 Phase 1。尚无任何真实模型后端（no real model backend exists yet）。

FuzzyAI 是一个 provider-agnostic（供应商无关）的概率决策运行时（probabilistic decision runtime）。它把语言模型变成可评估（evaluable）、可校准（calibratable）、可追踪（trackable）的语义概率决策组件。

要解决的问题：LLM 被接入程序逻辑时，人们常把一坨原始文本或一个未经审视的分数当作可信的概率。结果是：系统说不清一个分数意味着什么，无法复现过去的决策，也无法区分"模型不确定"与"程序应当拒绝行动"。FuzzyAI 为这一领域提供一个狭窄而确定性的内核。

它不是聊天框架，不是 agent 框架，也不仅仅是 structured-output（结构化输出）的包装层。

## 核心原则

> **LLM 处理语义不确定性。程序处理确定性策略。**
> The LLM handles semantic uncertainty. The program handles deterministic policy.

## Planned API（尚未实现）

以下接口是 **planned API（尚未实现）**。Phase 1 中不存在 `ai.bool` 或 `ai.choice` 入口。

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

- **概率（Probability）**：由某种 scoring strategy（打分策略）产生的决策分布。它并不自动等于现实世界中的正确率。
- **Certainty**：只描述该概率分布有多集中（熵 entropy、margin）。它是一个数学属性，不是正确性概率。我们从不把它称为 "confidence"。规范性约束：certainty 不得被称为 confidence（置信度）。
- **Predicted correctness（预测正确性）**：只有在有效校准之后才可表达。Phase 1 的每个结果都是 `predicted_correctness = None`。max softmax、logit、熵，或 LLM 自称"我有 95% 把握"，都不是 predicted correctness。

完整的规范性定义见[设计宪法](docs/design-constitution.md)。

## 当前已实现的内容（Phase 1）

Phase 1 交付的是契约（contracts），不是推理（inference）：

- `BoolDecision` 与 `ChoiceDecision` 决策规格（Phase 1 仅有的 primitives）
- 结果模型：`BoolResult`、`ChoiceResult`、`Certainty`
- `BackendCapabilities`（显式数据）与狭窄的 `Backend` Protocol
- `InferencePlan` 与 `RawEvidence` 抽象
- 确定性指纹系统（canonical JSON + SHA-256）
- 以 `FuzzyAIError` 为根的错误分类体系

Compiler 与所有真实后端均为未来工作。架构在纸面上成立，但尚未端到端接通。

```
DecisionSpec  --(Compiler, future)-->  InferencePlan  -->  Backend (Protocol)  -->  RawEvidence
                                                                                        |
                                                          (future scoring/calibration)  v
                                                                                  DecisionResult
```

## 开发

本项目自身的开发命令（需要 [uv](https://docs.astral.sh/uv/)）：

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

## 文档

- [docs/design-constitution.md](docs/design-constitution.md)：具有约束力的设计约束、术语表与规范性不变量。提出任何方案之前请先阅读。
- [docs/roadmap.md](docs/roadmap.md)：预期的阶段划分。不是时间表，也不是承诺。

## 非目标（Non-goals）

Phase 1 没有模型加载，没有 HTTP，没有 transformers/OpenAI/Anthropic/vLLM/SGLang 集成，没有自动路由，没有 decision graph，没有校准算法，也没有 dashboard、server、agent、RAG、数据库、telemetry 或 web UI。本项目不做任何 benchmark 或性能声明，也不声称支持任何模型。

## 许可证

MIT。见 [LICENSE](LICENSE)。
