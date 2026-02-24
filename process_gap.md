# Process Gap：当前实现全链路

---

## 阶段一：离线构建（`factory.py`）

### ① Agent C 分类

Agent C 在 `ALIGNMENT_JUDGE_PROMPT` 中判断一对 leaf 是 misaligned 的，并将 `relation_type` 设为 `"ProcessGap"`。触发条件根据 prompt 中的定义：

> Factual errors disrupt the discussion, or the expert lacks standardized procedures or falls into narrow narratives (tunnel vision), causing workflow/sequence misalignment.

### ② 创建 GapLink

这对 misaligned 的 leaf pair 被构建为一条 `GapLink`，此时 `assistance_payload` 是空字典 `{}`。

### ③ 调用 `PROCESS_GAP_PROMPT` 生成离线 payload

`generate_assistance_payload_for_link()` 根据 `relation_type == ProcessGap` 分发到 `PROCESS_GAP_PROMPT`。

LLM 接收的输入：
- Expert concept: path, label, description
- Researcher concept: path, label, description
- Misalignment reason（来自 Agent C 的 reason 字段）

LLM 生成的输出（存入 `link.assistance_payload`）：

```json
{
  "expected_steps": [
    {"label": "初始评估", "description": "...", "order": 1},
    {"label": "给药决策", "description": "...", "order": 2},
    {"label": "随访观察", "description": "...", "order": 3}
  ],
  "tunnel_vision_risks": [
    "给药决策 — 专家容易在此过度展开用药细节，忽略随访观察"
  ]
}
```

**注意：离线阶段只生成"预测的 workflow"和"哪些步骤可能 tunnel vision"，没有任何对话话术。**

---

## 阶段二：运行时状态管理（`app.py` + `schema.py`）

### ④ `DSAGState` 持有 `interview_timeline`

`schema.py` 中 `DSAGState` 有一个累积列表字段：

```python
interview_timeline: List[Dict[str, Any]] = field(default_factory=list)
```

### ⑤ 每轮对话后追加 timeline entry

`app.py` 的 `/api/dsag/analyze_turn` 处理完一轮后，把当前轮的信息追加进去：

```python
timeline_entry = {
    "turn_index": len(state.interview_timeline) + 1,
    "topic_label": expert_node.label if expert_node else "",
    "expert_leaf_id": analysis.located.best_expert_leaf_id or "",
    "researcher_leaf_id": analysis.located.best_researcher_leaf_id or "",
    "summary": f"Q: {researcher_question[:80]} | A: {expert_answer[:80]}",
}
state.interview_timeline.append(timeline_entry)
```

---

## 阶段三：运行时分析（`runtime.py`）

### ⑥ `analyze_turn()` 传入 `interview_timeline`

```python
analysis.assistance = self.generate_assistance(
    analysis.located.best_expert_leaf_id,
    analysis.selected_link,
    interview_timeline=interview_timeline,
)
```

### ⑦ `generate_assistance()` 的 ProcessGap 分支

这是最核心的运行时逻辑，执行以下 5 步：

| 步骤 | 做什么 | 输出 |
|------|--------|------|
| 1 | 把累积的 `interview_timeline` 塞进 payload | `payload["timeline"]` |
| 2 | **重复话题检测**：当前 topic label 是否已在之前 timeline 中出现 | `drift_alerts` 追加一条 |
| 3 | **跳步检测**：对比当前步骤 `order` 与 `expected_steps`，找出被跳过的步骤 | `drift_alerts` 追加一条 |
| 4 | **Tunnel vision 检测**：统计 timeline 中每个 topic 出现次数，若被标记为 tunnel vision risk 的 topic 出现 ≥2 次，触发警告 | `drift_alerts` 追加一条 |
| 5 | **未覆盖兄弟话题建议**：找到 expert leaf 的 siblings，排除已出现过的，展示前 3 个 | `drift_alerts` 追加一条 |

### ⑧ 跳过 polish

```python
# Skip polishing for ProcessGap (timeline data, not prose)
if assistance.relation_type == RelationType.PROCESS_GAP.value:
    return assistance
```

ProcessGap 的 payload 是结构化数据，不是自然语言文本，所以不走 polish 模型。

---

## 最终输出结构

```json
{
  "relation_type": "ProcessGap",
  "payload": {
    "expected_steps": [
      {"label": "初始评估", "description": "...", "order": 1},
      {"label": "给药决策", "description": "...", "order": 2},
      {"label": "随访观察", "description": "...", "order": 3}
    ],
    "tunnel_vision_risks": [
      "给药决策 — 专家容易在此过度展开用药细节"
    ],
    "timeline": [
      {"turn_index": 1, "topic_label": "给药决策", "expert_leaf_id": "...", "summary": "..."},
      {"turn_index": 2, "topic_label": "给药决策", "expert_leaf_id": "...", "summary": "..."}
    ],
    "drift_alerts": [
      "Topic '给药决策' has been discussed before (turn 1). Consider probing a different aspect.",
      "Skipped steps: 初始评估. Consider circling back.",
      "Tunnel vision warning: 给药决策 — 专家容易在此过度展开用药细节",
      "Related uncovered topics: 随访观察, 不良反应监测"
    ],
    "current_topic": "给药决策"
  },
  "followup_questions": []
}
```

---

## 全链路示意图

```
离线阶段 (factory.py)
  Agent C 标记 → ProcessGap
       ↓
  PROCESS_GAP_PROMPT → LLM 生成 {expected_steps, tunnel_vision_risks}
       ↓
  存入 link.assistance_payload
       ↓
═══════════════════════════════════════
运行时阶段 (app.py + runtime.py)

  每轮对话 →
       ↓
  app.py: 追加 timeline_entry 到 state.interview_timeline
       ↓
  runtime.py: locate_positions() → 定位到某个 expert_leaf
       ↓
  find_best_link() → 选中一条 ProcessGap 的 GapLink
       ↓
  generate_assistance():
    ├─ 复制离线 payload (expected_steps, tunnel_vision_risks)
    ├─ 注入 timeline (累积记录)
    ├─ 运行 4 种 drift detection → 生成 drift_alerts (纯文本)
    └─ 返回完整 payload
       ↓
  跳过 polish
       ↓
  返回给前端
```

---

## 当前局限

离线阶段生成了一份"参考答案"（expected workflow），运行时拿它和实际对话轨迹做对比，检测偏离，产出纯文本警告。但整个流程从头到尾没有生成任何一句研究者可以直接说出口的话——它只告诉研究者"出了什么问题"，不告诉研究者"该说什么"。
