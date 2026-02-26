# Process Gap：Runtime-Driven 改造方案

> 本文档是给 partner review 的完整技术方案。包含：设计理念、与旧方案的对比、详细流程定义、代码改动清单。

---

## 1. 为什么要改

### 旧方案的问题

旧方案在离线阶段让 LLM 凭 expert leaf 的 label + description **猜测**一份"预期工作流"（`expected_steps`），然后在运行时拿这个猜测去对比真实的对话记录，检测"跳步"和"tunnel vision"。

问题：

1. **Scaffold 是 LLM 幻觉**：它不知道专家实际的 workflow 长什么样。如果猜错（多了一个不存在的步骤），下游所有 drift detection 都是噪声。
2. **只告警不给话术**：检测到 drift 后只产出纯文本警告（如 "Skipped steps: 初始评估. Consider circling back."），不给研究者可以直接说出口的对话策略。
3. **用假的对比真的**：离线猜测 vs. 运行时真实数据，先天不匹配。

### 核心洞察

Process Gap 和其他 4 种 gap type 本质不同：

| | Lexical / Conceptual / Tacit / Scope | Process |
|---|---|---|
| 问题性质 | **静态**——两棵树之间的结构性错位，在 graph build 完就固定了 | **动态**——只在对话过程中才会显现 |
| 离线能做什么 | 很多——term mapping、analogy、probes、validate-pivot 都可以预生成 | **几乎无法预生成**——问题的定义依赖于"到目前为止发生了什么" |
| 运行时有什么 | 只需要 polish 润色 | **拥有全部所需信息**：真实对话历史 + 树结构 + 专家原话 |

结论：**Process Gap 应该完全 runtime-driven，不做任何离线预计算。**

---

## 2. 新方案定义

### 离线阶段

Agent C 将 GapLink 标记为 `ProcessGap`。**没有专用 prompt，没有 scaffold，`assistance_payload` 为空字典 `{}`。**

`PROCESS_GAP_PROMPT` 从 `factory.py` 中移除。`generate_assistance_payload_for_link()` 对 ProcessGap 类型直接返回 `{}`，跳过 LLM 调用。

### 运行时数据源

全部来自已有的、可靠的数据，无 LLM 假设：

| 数据源 | 是什么 | 提供什么 |
|--------|--------|----------|
| `interview_timeline` | 每轮对话后机械追加的记录列表 | 真实的 step→step 对话历史：哪些 topic 被讨论了、讨论了几次、什么顺序 |
| Expert tree structure | Agent A 生成的专家思维树 | 话题宇宙：当前 leaf 的兄弟节点（同 L2 下的相关 topics）、L2 parent category |
| 当前轮 `expert_answer` | 专家的原话 | Runtime LLM 生成 redirect 时可以引用专家的实际措辞 |
| 当前轮 `researcher_question` | 研究者的提问 | 上下文 |

### Drift Detection（三种，全部 rule-based，无 LLM）

#### Type 1: Repeated Topic

- **触发条件**：当前轮的 `topic_label` 在 `interview_timeline` 的历史记录中已出现过。
- **依赖**：纯 timeline 数据。与 tree structure 无关。
- **误报风险**：低。如果 embedding 把不同的回答匹配到同一个 leaf，那确实是在聊同一个话题。

#### Type 2: Tunnel Vision

- **触发条件**：同一个 `topic_label`（或同一个 L2 category）在最近连续或近连续的 ≥3 轮中出现，**且**该 leaf 存在至少 1 个未被 timeline 覆盖的兄弟节点。
- **依赖**：timeline 频率统计 + tree 的 sibling 结构。
- **为什么需要"兄弟节点未覆盖"条件**：如果所有 siblings 都已聊过，那专家深入某个 topic 是合理的（已完成广度，正在做深度）。只有"广度不足 + 深度过剩"才算 tunnel vision。

#### Type 3: Coverage Gap

- **触发条件**：当前 expert leaf 的兄弟节点中，存在从未在 timeline 中出现过的 leaf。
- **依赖**：tree 的 sibling 结构 + timeline 覆盖记录。
- **这不是 alert，而是 info**：Coverage Gap 不代表"出了问题"，它只是告诉研究者"这些相关话题还没聊到"。是否探索由研究者决定。

### Redirect Generation（仅在 drift 检测到时触发，runtime LLM call）

**当 Repeated Topic 或 Tunnel Vision 被检测到时**：

调用一次 runtime LLM，输入包含：

```
- drift_type: "repeated_topic" 或 "tunnel_vision"
- current_topic: 当前 topic label + description
- expert_answer: 专家本轮的原话（截断至前 500 字）
- timeline_summary: 最近 3-5 轮的 topic_label 序列
- unvisited_siblings: 未覆盖的兄弟节点的 label + description 列表
- misalignment_reason: Agent C 对这条 link 的 reason
```

LLM 生成一句自然的、引用了专家实际措辞的重定向话术。

**当只有 Coverage Gap（无 drift）时**：

不调用 LLM。只返回 coverage map（结构化数据）。

---

## 3. Runtime 完整流程

```
研究者提问，专家回答
        ↓
① locate_positions()
   embedding 匹配 → expert_leaf_X, researcher_leaf_Y
        ↓
② find_best_link()
   找到连接 X 和 Y 的 GapLink
        ↓
③ 判断 link.relation_type
   如果不是 ProcessGap → 走原有逻辑（复制离线 payload + polish）
   如果是 ProcessGap ↓
        ↓
④ 收集上下文
   ├─ interview_timeline（完整历史）
   ├─ expert_tree.get_siblings(expert_leaf_X)
   ├─ expert_answer, researcher_question
   └─ timeline 中已覆盖的 topic_labels 集合
        ↓
⑤ Rule-based Drift Detection
   ├─ Repeated Topic?
   │   当前 topic_label ∈ 历史 topic_labels → drift_type = "repeated_topic"
   │
   ├─ Tunnel Vision?
   │   当前 topic 在最近 ≥3 轮中出现 且 存在未覆盖兄弟 → drift_type = "tunnel_vision"
   │
   └─ Coverage Gap?（始终计算）
       未覆盖的兄弟节点列表
        ↓
⑥ 分支
   ├─ drift 检测到 → Runtime LLM call → 生成 redirect 话术
   │
   └─ drift 未检测到 → 只返回 coverage map
        ↓
⑦ 组装 Assistance payload
        ↓
⑧ 返回给前端（不走 polish — redirect 已经是 LLM 生成的自然语言）
```

---

## 4. Output 数据结构

### 检测到 drift 时

```json
{
  "relation_type": "ProcessGap",
  "payload": {
    "coverage": {
      "visited": ["误报太多", "选药"],
      "unvisited_siblings": ["报警麻木", "遗漏关键报警", "监测反应"],
      "coverage_ratio": "2/5"
    },
    "drift_detected": true,
    "drift_type": "tunnel_vision",
    "drift_detail": "'选药' appeared in turns 2, 3, 5. Siblings '报警麻木', '监测反应' not yet covered.",
    "redirect": "您刚提到选药时主要看肝肾功能，这个标准很实用。我也想了解一下，实际给药环节——比如给药速度、配伍禁忌——您通常怎么把控？"
  },
  "followup_questions": []
}
```

### 未检测到 drift 时

```json
{
  "relation_type": "ProcessGap",
  "payload": {
    "coverage": {
      "visited": ["误报太多"],
      "unvisited_siblings": ["报警麻木", "遗漏关键报警"],
      "coverage_ratio": "1/3"
    },
    "drift_detected": false,
    "drift_type": null,
    "drift_detail": null,
    "redirect": null
  },
  "followup_questions": []
}
```

---

## 5. 与其他 4 种 Type 的架构对比

|  | Lexical | Conceptual | Tacit | Scope | **Process (new)** |
|---|---|---|---|---|---|
| 问题性质 | 静态 | 静态 | 静态 | 静态 | **动态** |
| 离线 payload | term_mapping | analogy+scenario | probes+scenarios | validate+pivot | **空 `{}`** |
| 离线 LLM call | 有 | 有 | 有 | 有 | **无** |
| Runtime LLM call | polish | polish | polish | polish | **redirect generation（仅 drift 时）** |
| 依赖对话历史 | 否 | 否 | 否 | 否 | **是** |
| 跳过 polish | 否 | 否 | 否 | 否 | **是（redirect 已是自然语言）** |

---

## 6. 代码改动清单

### `factory.py`

| 改动 | 详情 |
|------|------|
| 移除 `PROCESS_GAP_PROMPT` | 删除整个 prompt 常量 |
| 修改 `generate_assistance_payload_for_link()` | 当 `relation_type == ProcessGap` 时直接 `return {}`，跳过 LLM 调用 |
| 修改 `prompt_map` | 移除 `ProcessGap` 条目 |

### `runtime.py`

| 改动 | 详情 |
|------|------|
| 新增 `PROCESS_GAP_REDIRECT_PROMPT` | Runtime prompt：输入 drift context + expert answer + siblings → 输出一句 redirect |
| 重写 `generate_assistance()` 的 ProcessGap 分支 | 移除旧的 scaffold 对比逻辑，替换为：收集 context → rule-based drift detection → 条件 LLM call |
| 新增 `_detect_process_drift()` 辅助函数 | 封装三种 drift detection 规则逻辑 |
| 新增 `_generate_process_redirect()` 辅助函数 | 封装 runtime LLM call 逻辑 |
| 更新 `Assistance` docstring | ProcessGap payload shape 从旧结构改为新结构 |
| 移除 `skip polish for ProcessGap` 的特殊判断 | 旧代码跳过 polish 因为旧 payload 是结构化数据。新方案中 redirect 已经是 LLM 生成的自然语言，同样不需要 polish，但原因不同。保持跳过即可。 |

### `app.py`

| 改动 | 详情 |
|------|------|
| 无改动 | Timeline 追加逻辑不变。`analyze_turn` 调用不变。 |

### `schema.py`

| 改动 | 详情 |
|------|------|
| 无改动 | `GapLink.assistance_payload` 本来就是 `Dict[str, Any]`，空字典 `{}` 合法。`DSAGState.interview_timeline` 不变。 |

---

## 7. 已知局限

1. **失去了时序判断能力**：旧 scaffold 有 `order` 字段可以判"跳步"。新方案只能判"未覆盖"，不能判"顺序错了"。这是有意的 trade-off——用不可靠的顺序判断不如不判断。
2. **Tunnel vision 没有先验标注**：旧 scaffold 的 `tunnel_vision_risks` 标注了哪些 topic 容易卡住。新方案对所有 topic 一视同仁，纯靠频率。通过设置合理阈值（≥3 轮 + 有未覆盖兄弟）来控制误报。
3. **Coverage 粒度为 leaf 级别**：如果 expert tree 的 leaf 太粗（某个 L2 下只有 1-2 个 leaf），coverage map 作用有限。这取决于 Agent A 的 tree 生成质量，不是 Process Gap 本身的问题。
4. **Process Gap 监控只在 ProcessGap link 被选中时才触发**：如果专家在 ConceptualGap 话题上 tunnel vision，Process Gap 监控不会介入。未来可以考虑将 drift detection 抽象为 global monitoring layer，但当前先保持 type-specific。
5. **Runtime LLM call 增加延迟**：当 drift 被检测到时，会有一次额外的 LLM 调用（使用 light/polish model）。无 drift 时无额外调用。

---

## 8. Advisory Principle

所有 alerts 和 redirects 都是建议，不是指令。研究者可以忽略任何提示。熟练的访谈者可能故意允许专家深入某个话题——系统不应 paternalistic。
