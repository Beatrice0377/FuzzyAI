[English](README.md) | **简体中文**

# Probvenance

**状态：早期开发阶段（Status: early development）。** 确定性内核（deterministic core）、Bool 垂直切片（vertical slice，真实本地 Hugging Face 后端），以及 direct categorical Choice 推理（仅限 closed-set、single-label、single-token 打分标签路径）均已实现。Phase 4B winner-correctness 评估基础（evaluation cohort / dataset、Brier、exact log loss、companion diagnostics、等宽可靠性分箱、ECE-form binned absolute-gap aggregate）已实现。`CalibrationProfile` 身份基座（identity foundation）同样已实现：档案 artifact 会提交其 binding、ground-truth 语义、target、input-score、method、fitted parameters 与训练数据集来源；遇到具体的 taxonomy 矛盾或 binding 不完全匹配时一律 fail closed。已实现一个受支持的标量拟合方法：`fit_l2_logistic_selected_probability`，它把选中概率经 L2 正则化逻辑映射拟合成 winner correctness。离线档案应用（offline profile application）与校准后评估基座（post-calibration evaluation foundation）已实现：可以把一个精确匹配的 profile 应用到一份兼容的 evaluation dataset，产出一个不可变的 predicted-winner-correctness artifact，该 artifact 逐行同时记录推导出的 winner-correctness 目标标签与 profile 产出的分数；随后由校准后 Brier、exact log loss、companion diagnostics、等宽可靠性（equal-width reliability）以及 binned absolute-gap aggregate 消费该 artifact。已实现显式运行时链接档案应用（explicit runtime-linked profile application）：调用者可以把一个精确匹配的 profile 应用到一份未校准的运行时 `Evaluation`，得到携带 `predicted_correctness` 的校准结果，以及镜像同一 profile 身份的 trace。`CalibrationProfile` 还具有带版本号的规范化 JSON 序列化（serialization）与经过身份校验的加载器（identity-verified loader）：加载器会重建嵌套的 binding 与 ground-truth 语义身份，并逐一重新校验每个 fingerprint，而不是信任文档中的声明。已实现精确内容寻址的目录存储（exact content-addressed directory store）：它把 profile 持久化到仅由其 fingerprint schema 版本与精确 fingerprint 推导出的路径下，检索时要求同时提供这两个值，并通过身份校验加载器恢复工件、把请求的身份作为独立的 expected pin，且不做任何匹配或回退。已实现显式运行时档案选择（explicit runtime profile selection）：`select_calibration_profile_for_runtime` 接收一份未校准的 evaluation 与一个由调用者显式提供的候选 profile 元组，返回唯一合格的 profile；没有任何合格者时抛出显式的 no-eligible 错误，多于一个互不相同的合格 profile 时抛出歧义错误，且不做任何 tie-break。选择是授权（authorization），不是推荐（recommendation）：它与存储无关、不做应用，也不以质量指标或方法支持度作为偏好。已实现显式内存档案目录（explicit in-memory profile catalog）与非权威的运行时发现（non-authoritative runtime discovery）：`CalibrationProfileCatalog` 由调用者显式提供的 profile 元组构建，`discover_calibration_profile_references_for_runtime` 返回确定性的精确 profile 引用（reference），其发现元数据与选择阶段使用同一套运行时合格性投影，但发现本身不做授权、不加载、不选择、也不应用任何 profile。目录还具有精确的快照身份（snapshot identity，`canonical_payload` / `fingerprint`）以及确定性的、经身份校验的序列化与加载（`serialize_calibration_profile_catalog` / `load_calibration_profile_catalog`）：快照提交有序的引用集合以及 Binding 与 target/input 发现投影，加载时重新计算规范化 payload 与目录 fingerprint，而不是信任文档中嵌入的哈希。目录快照是非权威的发现元数据：其 fingerprint 证明的是快照自身的身份，而不是被引用 profile 仍然存在、存储工件完整，或快照元数据仍与真实 profile 一致。自动运行时档案选择（automatic profile selection）、档案注册表（registry）、基于 binding 的查找（lookup）、目录文件系统/数据库存储（catalog filesystem/database store）、存储枚举（store enumeration）、质量排序（quality ranking）与签名分发（signed distribution）均不存在；任何未经调用者显式校准的运行时结果，其 `predicted_correctness` 仍为 `None`。弃权（abstention）与其他所有 Choice 策略尚未实现。

Probvenance 是一个 provider-agnostic（供应商无关）的概率决策运行时（probabilistic decision runtime）。它把语言模型变成可评估（evaluable）、可校准（calibratable）、可追踪（trackable）的语义概率决策组件。

要解决的问题：LLM 被接入程序逻辑时，人们常把一坨原始文本或一个未经审视的分数当作可信的概率。结果是：系统说不清一个分数意味着什么，无法复现过去的决策，也无法区分"模型不确定"与"程序应当拒绝行动"。Probvenance 为这一领域提供一个狭窄而确定性的内核。

它不是聊天框架，不是 agent 框架，也不仅仅是 structured-output（结构化输出）的包装层。

## 核心原则

> **LLM 处理语义不确定性。程序处理确定性策略。**
> The LLM handles semantic uncertainty. The program handles deterministic policy.

## 当前可用：Bool 与 Choice 垂直切片

目前有两条端到端路径：Bool 切片，以及实验性的 direct categorical Choice 切片。一个轻量的 `Probvenance` facade 按固定顺序编排每条流水线，自身不添加任何回退（fallback）。自 Phase 2A.1 起，Bool 切片还会在每条 trace 上报告 scoring-validity 诊断（scoring-validity diagnostics）与执行指纹（execution fingerprint）：

```
BoolDecision -> BoolCompiler -> InferencePlan -> TransformersBackend
   -> RawEvidence -> assemble_bool_probability -> BoolResult -> DecisionTrace
```

后端是可选附加包（optional extra）。基础包没有任何运行时依赖；只有安装附加包才会引入 `torch` 与 `transformers`：

```bash
uv sync --extra transformers    # 依赖组：torch>=2.7, transformers>=4.53
```

```python
from probvenance import BoolDecision
from probvenance.backends.transformers import TransformersBackend
from probvenance.runtime import Probvenance

backend = TransformersBackend(
    "Qwen/Qwen3-0.6B",  # 一个本地 HF causal LM
    # 否则 Qwen3 会先输出推理块；见下方说明。
    chat_template_kwargs={"enable_thinking": False},
)
ai = Probvenance(backend=backend)

evaluation = ai.evaluate_with_trace(
    BoolDecision(
        question="Is this a delivery issue?",
        context="My package never arrived.",
    )
)

print(evaluation.result.probability_true)
print(evaluation.trace.scoring_diagnostics.verbalizer_mass)
print(evaluation.trace.execution_fingerprint)
```

`ai.evaluate(decision)` 是同一条流水线，只返回 `BoolResult`（`probability_true`、`certainty`、`predicted_correctness`、`trace_id`）。

`ai.evaluate_with_trace(decision)` 返回一个 `Evaluation`（结果加上它的 `DecisionTrace`）。trace 记录了 decision 与 plan 的指纹（fingerprint）、scoring strategy、doctrine id、解析出的 verbalizer token id、原始证据（raw evidence）、input fingerprint、backend 类型、延迟（latency）、`ScoringDiagnostics` 与 execution fingerprint。该 trace 是 replay-oriented provenance（面向回放的溯源）：它记录了未来回放所需的信息，但并不快照 backend 或 tokenizer 的代码，因此严格可回放性（strict replayability）仍是未决问题。

`probability_true` 是什么：一个有条件的、受限（restricted）的概率，`P(True | next token is one of the two scored verbalizer tokens)`，在恰好两个 verbalizer-token logit 上以数值稳定的 two-way softmax 计算，等于 `sigmoid(l_true - l_false)`。它无法告诉你模型是否本来就打算在这些候选项之间做出选择。它不是什么：不是全词表（full-vocabulary）概率，不是现实世界事件概率，不是预测准确率，也不是校准后的数值。`calibrated` 恒为 `False`，`predicted_correctness` 恒为 `None`。

由于受限数值本身无法显示模型是否处在决策点，每条 trace 还携带一个独立的词表级量 `trace.scoring_diagnostics.verbalizer_mass`：`P(next token is one of the two scored verbalizer tokens)`，其稳定对数形式为 `log_verbalizer_mass = logsumexp([l_true, l_false]) - logsumexp(all_vocab_logits)`。解读规则：`probability_true = 0.75` 且 `verbalizer_mass = 0.000001` 意味着模型在内部更偏好 `yes` 而非 `no`，但它几乎肯定不会输出两者中的任何一个；因此除非同时说明其受限性质，`0.75` 不得被读作「有 75% 的倾向回答 yes」。诊断不携带任何裁决：Phase 2A.1 不施加任何阈值，也不做自动拒绝。

后端加载一个本地 causal LM，在 `eval()` 与 `torch.inference_mode()` 下运行，优先使用 CUDA 并回退到 CPU，且只读取最后一个位置的 logits。它从不调用 `generate()`，从不解析生成的文本。verbalizer 默认为 `yes`/`no`，且各自必须在真实的 chat-template 前缀之后解析为恰好一个、互不相同的打分 token；否则抛出 `VerbalizerError`。没有静默回退，没有截断，没有多 token logit 求和，也没有采样回退。

`probability_true` 只有在模型确实处在决策位置时才有意义，也就是说它的下一个 token 真的必须是两个 verbalizer 之一。某些模型（thinking / reasoning 模型）会先输出推理块：如果 chat-template 的渲染模式不对，模型概率最高的 token 会是它的推理开标签，两个 verbalizer token 都落在分布极尾部，两路 softmax 于是把尾部噪声重归一化成一个看起来合理的数字。上面的示例因此使用 `chat_template_kwargs={"enable_thinking": False}` 渲染。诊断信息让这一点可见：实测差异记录在 [docs/claims.md](docs/claims.md)，同一个 plan、同一个问题，关闭 thinking 时 `P(True) = 0.9988`、`verbalizer_mass = 0.954228`（top token `'yes'`，id 9693，概率 0.953119）；开启 thinking 时 `P(True) = 0.5116`、`verbalizer_mass = 0.000000`（top token `'<think>'`，id 151667，概率 0.999699）。Phase 2A.1 不会自动检测这种情况，也不会自动拒绝任何结果，因为没有实验能支持一个跨模型、tokenizer、chat template、verbalizer 与 prompt 都稳定的阈值；而「拒绝作答」属于尚不存在的 policy 行为。

每次评估携带四个互不相同的身份标识，各自回答不同的问题：

- decision fingerprint（决策指纹）：正在被判断的是哪个语义问题
- plan fingerprint（计划指纹）：编译器产出了什么
- execution fingerprint（执行指纹）：该 plan 实际运行时所处的执行环境与渲染配置
- trace id：这是哪一次单独的执行

此外，每个被评估的概率还可追溯到一个显式的 probability formulation identity（概率表述身份）与 formulation-family identity（表述族身份），见 `docs/probability-semantics-identity.md`；两者都不包含模型、tokenizer 或输入。

execution fingerprint 的 payload 覆盖：plan fingerprint、backend 类型、backend 实现版本、模型标识、模型 revision、tokenizer 标识、tokenizer revision、运行时版本、dtype、渲染配置（rendering config）、input fingerprint，以及解析出的 positive/negative token id。同一个 plan 运行两次，会得到两个 trace id 与一个 execution fingerprint。影响概率语义的渲染配置（例如模型的 thinking 模式）会进入 execution fingerprint 与 trace，但刻意不进入与 provider 无关的 `InferencePlan`。

decision 的 context 只会被渲染进 user prompt 作为证据（evidence），永远不会进入 system prompt。这是一条结构性放置规则，不是 prompt-injection（提示注入）安全性声明。

`BoolCompiler` 与 `ChoiceCompiler` 相互排斥：`BoolCompiler` 会以 `UnsupportedDecisionError` 拒绝 `ChoiceDecision`，`ChoiceCompiler` 则拒绝 `BoolDecision`。实验性的 Choice 运行时见下文「当前已实现的内容」中的 Phase 2B direct categorical Choice 路径。

Phase 2A.2 用这条切片自带的诊断做了一次语义信号验证实验（semantic signal validation）：二值打分位置到底携不携带语义信号，测量结果里哪些部分来自模型、哪些来自 doctrine 与 label family。它在固定条件下探测了三个本地 causal LLM（每条 probe 一次前向传播，没有 ground truth），并如实记录观察到的现象。实验记录位于 [experiments/semantic_signal/REPORT.md](experiments/semantic_signal/REPORT.md)：那是一份实验记录，不是能力声明，也不是 benchmark。这一轮产出的一项具体修复是诊断侧的：`src/probvenance/diagnostics.py` 中的上界钳制（overshoot clamp）改成了相对容差（relative tolerance），见 [docs/claims.md](docs/claims.md)。

## Planned API（尚未实现）

下面的便捷接口仍是 **planned API（尚未实现）**。不存在 `ai.bool` 或 `ai.choice` 入口；当前可用的调用是 `ai.evaluate(BoolDecision(...))` 与 `ai.evaluate(ChoiceDecision(...))`。

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

- **概率（Probability）**：由某种 scoring strategy（打分策略）产生的决策分布。它并不自动等于现实世界中的正确率。当前 Bool 概率是在两个 verbalizer-token logit 上的受限（条件）概率，不是全词表分布；trace 中的 `verbalizer_mass` 是与之独立的词表级伴随量。
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
- 以 `ProbvenanceError` 为根的错误分类体系

Phase 2A Bool 路径：

- `ScoringDoctrine` 与默认 doctrine `BINARY_SEMANTIC_JUDGMENT_V1`
- `BoolCompiler`（纯规划器：它从不计算概率）
- `assemble_bool_probability`（binary token logits 上的 two-way softmax）
- `TransformersBackend`（可选 `transformers` 附加包，只读 logits）
- `DecisionTrace` / `build_decision_trace`（其 `trace_id` 从不派生自任何指纹）
- 轻量的 `Probvenance` / `Evaluation` facade

Phase 2A.1 的 scoring-validity 增量：

- `ScoringDiagnostics` / `diagnose_bool_evidence`（词表级统计量：`verbalizer_mass`、top token id、概率与尽力而为的解码文本；不设阈值、不下裁决）
- `DecisionTrace` 上的 execution fingerprint（同一个 plan 的两次执行：两个 trace id，一个 execution fingerprint）

```
BoolDecision --(BoolCompiler)--> InferencePlan --(TransformersBackend)--> RawEvidence
                                                                             |
                              assemble_bool_probability     (calibration: still future)
                                                                             v
                                                BoolResult + DecisionTrace (via Probvenance)
```

Phase 2B direct categorical Choice 路径：

- `ChoiceCompiler` 与 doctrine `CATEGORICAL_SEMANTIC_JUDGMENT_V1`，配合带版本的 `categorical-labels-v1` 标签方案（label scheme）
- `CandidateLabelMapping`（语义候选 → 打分标签，刻意不携带 token id，使 plan 保持 provider-independent）
- `assemble_choice_probability`（候选 logits 上的 N-way softmax）与 `ChoiceScoringDiagnostics`（`scoring_label_mass`、top token、每个候选的 token 概率）
- `TransformersBackend` 中的 N 路打分标签解析与校验
- `ChoiceResult` 以语义候选名称为键，平局按语义候选顺序裁决

```
ChoiceDecision --(ChoiceCompiler)--> InferencePlan --(TransformersBackend)--> RawEvidence
                                                                                 |
                        assemble_choice_probability    (calibration: still future)
                                                                                 v
                                             ChoiceResult + DecisionTrace (via Probvenance)
```

该路径是实验性的。它是 closed-set 的，假定调用方提供互斥候选，只支持 single-label 决策，要求每个打分标签恰好是一个 token，仅在较小 N 上做过研究，并且是**未校准**的（`predicted_correctness` 始终为 `None`）。它没有任何 open-set 保证：当候选集遗漏了真实主题时，模型仍会在集合内作答，而 `scoring_label_mass` 检测不到这一点。

其他 Choice 策略（one-vs-rest、sampling、multi-token 打分标签）、校准与弃权仍是未来工作。

## 证据与主张（Evidence and claims）

性能（performance）、质量、校准（calibration）与 provider 支持相关的声明，统一按证据状态（evidence status）记录在 [docs/claims.md](docs/claims.md) 中。本项目目前不做任何 benchmark、性能或模型支持声明：Bool 路径可以对着本地 Hugging Face causal LM 运行，但没有任何模型对着 ground truth 被评估过。Phase 2A.2 针对二值打分位置运行了一次语义信号验证实验（semantic signal validation），探测了三个本地 Hugging Face causal LM；其记录位于 [experiments/semantic_signal/REPORT.md](experiments/semantic_signal/REPORT.md)，是一份记录既定条件下所观察到的信号行为的实验记录，不是能力声明，也不是 benchmark。假设（hypotheses）与路线图条目在主张登记册（claims register）中均被如实标注，不作为已实现的能力呈现。Phase 2B 针对 direct categorical Choice 运行了一次实验，覆盖标签排列（label permutation）、无关候选新增、description 改写、分类体系重叠（taxonomy overlap）与 out-of-set 探测，使用一个本地模型；Phase 2B.1 在完全相同的冻结 case set 上把该实验复现到第二个 model family；其记录位于 [experiments/choice_signal/REPORT.md](experiments/choice_signal/REPORT.md)。

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
