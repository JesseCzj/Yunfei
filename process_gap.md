# Process Gap：Runtime-Driven 实现文档

> 本文档记录 Process Gap 的**实际实现**。与代码完全对应。

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

## 2. 当前实现

### 离线阶段（`factory.py`）

Agent C 将 GapLink 标记为 `ProcessGap`。**没有专用 prompt，没有 scaffold，不调用 LLM。**

- `PROCESS_GAP_PROMPT` 已从 `factory.py` 中移除（替换为注释）。
- `generate_assistance_payload_for_link()` 在 `relation_type == ProcessGap` 时直接返回 `{"misalignment_reason": reason}`，跳过 LLM 调用。`misalignment_reason` 来自 Agent C 的对齐判断，存储在 payload 中供 runtime redirect prompt 消费。
- `prompt_map` 字典中不再包含 `ProcessGap` 条目。

### 运行时数据源

全部来自已有的、可靠的数据，无 LLM 假设：

| 数据源 | 是什么 | 提供什么 |
|--------|--------|----------|
| `interview_timeline` | 每轮对话后机械追加的记录列表 | 真实的 step→step 对话历史：哪些 topic 被讨论了、讨论了几次、什么顺序 |
| Expert tree structure | Agent A 生成的专家思维树 | 话题宇宙：当前 leaf 的兄弟节点（同 L2 下的相关 topics）、L2 parent category |
| 当前轮 `expert_answer` | 专家的原话 | Runtime LLM 生成 redirect 时可以引用专家的实际措辞 |
| 当前轮 `researcher_question` | 研究者的提问 | 上下文 |
| `misalignment_reason` | Agent C 在离线阶段生成的对齐理由 | 存储在 `assistance_payload` 中，runtime 读取后注入 redirect prompt |

### Drift Detection（两种 alert + 一种 info，全部 rule-based，无 LLM）

两种 drift alert 按优先级顺序检测，**只取第一个命中的**（互斥）。Coverage Gap 是 info，始终计算但不触发 redirect。

#### Type 1: Repeated Topic（drift alert，优先级最高）

```python
topic_count = sum(1 for v in visited_labels if v == current_label)
if topic_count >= 2:
    drift_type = "repeated_topic"
```

- **触发条件**：当前 topic_label 在 `interview_timeline` 历史中已出现 **≥2 次**（即当前是第 3+ 次讨论此话题）。
- **注意**：timeline 在 `analyze_turn()` 返回后才追加新条目（见 `app.py`），所以 `visited_labels` 不包含当前轮。`topic_count >= 2` 意味着历史中有 2 条记录 + 当前这一轮 = 至少第 3 次。

#### Type 2: Tunnel Vision（drift alert）

```python
if not drift_type and len(timeline) >= 4 and unvisited:
    recent_ids = [entry.get("expert_leaf_id", "") for entry in timeline[-4:]]
    if len(set(recent_ids)) == 1 and recent_ids[0]:
        drift_type = "tunnel_vision"
```

- **触发条件**：`timeline` 中最近 4 轮全部映射到**同一个 expert leaf**，**且**存在至少 1 个未覆盖的兄弟节点。
- **为什么需要兄弟条件**：如果所有 siblings 都已聊过，专家深入某个 topic 是合理的（广度已完成，正在做深度）。只有"广度不足 + 深度过剩"才算 tunnel vision。

#### Coverage Gap（info，不触发 redirect）

- **始终计算**：无论是否检测到 drift，coverage 信息都会包含在 payload 中。
- **内容**：已覆盖的 topic 列表、未覆盖的兄弟节点列表、覆盖率。
- **性质**：这是信息展示，不是 alert。前端可以渲染为进度条或 checklist，研究者自行决定是否探索未覆盖话题。
- **不触发 LLM**：与 Repeated Topic / Tunnel Vision 不同，Coverage Gap 不会调用 LLM 生成 redirect。

### Redirect Generation（`_generate_process_redirect()`）

**仅当 Repeated Topic 或 Tunnel Vision 被检测到时**，调用一次 runtime LLM（`_build_light_llm()`），使用 `PROCESS_GAP_REDIRECT_PROMPT`。

输入包含：

| 变量 | 来源 | 处理 |
|------|------|------|
| `drift_type` | drift detection 结果 | 原样传入 |
| `drift_detail` | drift detection 生成的描述 | 原样传入 |
| `current_topic_label` | `expert_node.label` | 原样传入 |
| `current_topic_description` | `expert_node.description` | 原样传入 |
| `expert_answer` | 当前轮专家原话 | 截断至 500 字 |
| `unvisited_siblings_text` | `expert_tree.get_siblings()` 中未覆盖的节点 | 取前 4 个，格式化为 `- label: description` |
| `misalignment_reason` | `selected_link.assistance_payload["misalignment_reason"]` | Agent C 的对齐理由 |
| `timeline_summary` | `interview_timeline` 的最近 4 条 | 格式化为 `Turn N: [topic] summary` |

**Prompt 要求 LLM**：
1. 引用专家的实际措辞
2. 先肯定专家刚说的内容
3. 用探索性问题引入未覆盖话题
4. 保持 1-2 句，自然对话风格
5. 不使用专家未用过的术语

**当未检测到 drift 时**：不调用 LLM，`redirect` 为 `null`。

### Coverage Analysis（始终计算）

无论是否检测到 drift，都会返回 coverage 信息：

```python
coverage_info = {
    "visited": list(dict.fromkeys(visited_labels)),   # 去重、保序
    "unvisited_siblings": [s.label for s in unvisited[:5]],  # 最多 5 个
    "coverage_ratio": f"{visited_count}/{total_siblings}",
}
```

### Polish 步骤

ProcessGap 跳过 `_polish_assistance()`——redirect 已经是 LLM 在完整上下文下生成的自然语言。

```python
# analyze_turn() 内
if analysis.assistance.relation_type != RelationType.PROCESS_GAP.value:
    analysis.assistance = self._polish_assistance(...)
```

`_POLISH_RULES_BY_TYPE` 字典中 ProcessGap 条目已替换为注释。

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
④ Coverage Analysis（始终执行）
   ├─ covered_ids = timeline 中所有 expert_leaf_id
   ├─ siblings = expert_tree.get_siblings(current_leaf)
   ├─ unvisited = siblings 中不在 covered_ids 中的
   └─ coverage_ratio = (total - unvisited) / total
        ↓
⑤ Drift Detection（按优先级，互斥，仅 2 种 alert）
   ├─ Repeated Topic?
   │   current_label 在 visited_labels 中出现 ≥2 次 → drift_type = "repeated_topic"
   │
   └─ Tunnel Vision?（仅前项未触发时）
       最近 4 轮全部同一个 expert_leaf 且有未覆盖兄弟 → drift_type = "tunnel_vision"
        ↓
⑥ 分支
   ├─ drift 检测到 → _generate_process_redirect() → LLM 生成 redirect 话术
   │
   └─ drift 未检测到 → redirect = null（coverage info 仍然返回）
        ↓
⑦ 组装 payload = {coverage, drift_detected, drift_type, drift_detail, redirect}
        ↓
⑧ 返回给前端（跳过 polish）
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
    "drift_detail": "The last 4 turns all discuss the same concept. Consider broadening the scope.",
    "redirect": "您刚提到选药时主要看肝肾功能，这个标准很实用。我也想了解一下，实际给药环节——比如给药速度、配伍禁忌——您通常怎么把控？"
  }
}
```

### 未检测到 drift 时（coverage info 仍返回）

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
  }
}
```

---

## 5. 与其他 4 种 Type 的架构对比

|  | Lexical | Conceptual | Tacit | Scope | **Process** |
|---|---|---|---|---|---|
| 问题性质 | 静态 | 静态 | 静态 | 静态 | **动态** |
| 离线 payload | term_mapping | analogy+scenario | probes+scenarios | validate+pivot | **`{"misalignment_reason": ...}` （仅存 reason）** |
| 离线 LLM call | 有 | 有 | 有 | 有 | **无** |
| Runtime LLM call | polish | polish | polish | polish | **redirect generation（仅 drift 时）** |
| 依赖对话历史 | 否 | 否 | 否 | 否 | **是** |
| 跳过 polish | 否 | 否 | 否 | 否 | **是（redirect 已是自然语言）** |

---

## 6. 代码改动清单（已完成）

### `factory.py`

| 改动 | 详情 |
|------|------|
| 移除 `PROCESS_GAP_PROMPT` | 37 行 prompt 替换为 2 行注释 |
| 修改 `generate_assistance_payload_for_link()` | ProcessGap 在 `prompt_map` 之前返回 `{"misalignment_reason": misalignment_reason}`，跳过 LLM |
| 修改 `prompt_map` | 移除 `ProcessGap` 条目 |

### `runtime.py`

| 改动 | 详情 |
|------|------|
| 新增 `PROCESS_GAP_REDIRECT_PROMPT` | 位于 `_parse_json_from_text()` 之后、Result Data Classes 之前 |
| 新增 `_generate_process_redirect()` 方法 | RuntimeEngine 的实例方法，封装 LLM redirect 生成 |
| 重写 ProcessGap 分支 | `generate_assistance()` 内：coverage analysis → drift detection（2 种 alert）→ conditional redirect |
| 更新 `generate_assistance()` 签名 | 新增 `expert_answer` 和 `researcher_question` 参数 |
| 更新 `analyze_turn()` 调用 | 传递 `expert_answer` 和 `researcher_question` 给 `generate_assistance()` |
| 更新 `analyze_turn()` polish 逻辑 | ProcessGap 跳过 `_polish_assistance()` |
| 更新 `Assistance` docstring | ProcessGap payload shape 改为新结构 |
| 更新 `_POLISH_RULES_BY_TYPE` | ProcessGap 条目替换为注释 |

### `app.py`

| 改动 | 详情 |
|------|------|
| 无改动 | Timeline entry 结构不变，字段名对齐 |

### `schema.py`

| 改动 | 详情 |
|------|------|
| 无改动 | `GapLink.assistance_payload` 为 `Dict[str, Any]`，`{"misalignment_reason": ...}` 合法 |

---

## 7. 已知局限

1. **失去了时序判断能力**：旧 scaffold 有 `order` 字段可以判"跳步"。新方案只能判"未覆盖"，不能判"顺序错了"。这是有意的 trade-off——用不可靠的顺序判断不如不判断。
2. **Tunnel vision 没有先验标注**：旧 scaffold 的 `tunnel_vision_risks` 标注了哪些 topic 容易卡住。新方案对所有 topic 一视同仁，纯靠频率 + 兄弟节点覆盖率。
3. **Coverage 粒度为 leaf 级别**：如果 expert tree 的 leaf 太粗（某个 L2 下只有 1-2 个 leaf），coverage map 作用有限。这取决于 Agent A 的 tree 生成质量。
4. **Process Gap 监控只在 ProcessGap link 被选中时才触发**：如果专家在 ConceptualGap 话题上 tunnel vision，Process Gap 监控不会介入。
5. **Runtime LLM call 增加延迟**：当 drift 被检测到时，一次额外 LLM 调用（使用 `_build_light_llm()`）。无 drift 时无额外调用。

---

## 8. Advisory Principle

所有 alerts 和 redirects 都是建议，不是指令。研究者可以忽略任何提示。熟练的访谈者可能故意允许专家深入某个话题——系统不应 paternalistic。
