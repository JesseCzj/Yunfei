# demo_flask

一个面向跨学科访谈（HCI + 专家领域）的访谈辅助系统。前端提供访谈界面与交互，后端基于 DSAG（Dual‑Tower Semantic Alignment Graph）构建知识图谱与运行时辅助。

> 重点目录：
> - `dsag/`：**所有后端核心细节**
> - `static/` + `templates/`：**前端细节**
> - `llm_backend.py`：旧版 demo（可忽略）

---

## 目录结构

```
demo_flask/
  app.py                  # Flask 服务入口（API + session 管理）
  dsag/                   # DSAG 后端核心模块
  static/                 # 前端 JS/CSS
  templates/              # 前端 HTML 模板
  visualize_dsag.py       # DSAG 离线构建 + 可视化
  test_dsag.py            # DSAG smoke test
  questionnaire.docx      # 研究者问卷（仅给 Researcher Tree）
  requirements.txt
  .env
```

---

## 快速开始

### 1) 安装依赖

```
pip install -r requirements.txt
```

### 2) 配置 .env

文本生成与向量模型使用不同 base_url/密钥（当前配置已支持）：

```
LLM_PROVIDER=openai

# 文本生成（qwen3-max）
OPENAI_API_KEY=YOUR_QWEN_API_KEY
OPENAI_BASE_URL=YOUR_QWEN_BASE_URL
OPENAI_MODEL=qwen3-max
OPENAI_MODEL_GRAPH=qwen3-max
OPENAI_MODEL_RUNTIME=qwen3-max
OPENAI_MODEL_POLISH=qwen-turbo

# 向量模型（text-embedding-3-small）
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EBD_API_KEY=YOUR_EMBEDDING_API_KEY
OPENAI_EBD_BASE_URL=YOUR_EMBEDDING_BASE_URL
```

### 3) 运行 Flask

```
python app.py
```

访问：`http://localhost:5000`

### 4) 离线构建与可视化 DSAG

```
python visualize_dsag.py
```

会生成：
- `dsag_output.json`
- `dsag_alignments.json`
- `dsag_visualization.html`

---

## DSAG 构建流程（重点）

```mermaid
flowchart TD
    input[Input] --> agentA[AgentA_ExpertTree]
    input --> agentB[AgentB_ResearcherTree]
    agentA --> agentC[AgentC_Alignments]
    agentB --> agentC
    agentC --> alg[BuildGapLinks]
    alg --> templates[BridgeTemplates]
    templates --> graph[DSAGGraph]
    graph --> embed[EmbeddingIndex]
    embed --> runtime[RuntimeAssistance]
```

### 1) Schema（`dsag/schema.py`）

**DSAGNode**
- `id`, `tower`, `layer`, `label`, `description`, `parent_id`, `children_ids`
- `attributes`：**Tacit Knowledge Facets**（隐性知识切面：决策启发式 / 环境线索 / 数据特征 / 价值权衡）
- `aliases`：同义词/别称

**GapLink**
- `expert_leaf_id`, `researcher_leaf_id`
- `aligned_path_expert` / `aligned_path_researcher`
- `lca_layer`, `expert_lca_id`, `researcher_lca_id`
- `relation_type`（五种 gap 类型）
- `bridge_templates`, `checkout_template`
- `weight`

**RelationType（沟通错位类型）**
- `LexicalGap` / `ConceptualGap` / `TacitGap` / `ScopeGap` / `ProcessGap`

**Alignment**
- `NodeAlignment`: (expert_node_id, researcher_node_id, is_aligned, reason, semantic_similarity)
- `TreeAlignments`: leaf_alignments + concept_alignments

### 2) Factory（`dsag/factory.py`）

**Agent A（Expert Tree）**
- 根据 topic + expert_bg 生成专家树
- `is_intuition=true` 的叶子会生成 `attributes`（Tacit facets）

**Agent B（Researcher Tree）**
- 读取 `questionnaire.docx`，提取问题为 Researcher 叶子节点
> 重要：**questionnaire 只提供给 Researcher Tree，不给 Expert Tree**

**Agent C（Alignment Judge）**
- 输出对齐/错位的 leaf/concept pairs

**GapLink 构建**
- 只对 `misaligned` pairs 生成 GapLink
- 计算 LCA、冲突分支、relation_type、weight

**Bridge Templates + Checkout**
- `generate_bridge_templates_for_link()` 用 LLM 填充三层桥接与 checkout 问题

**Fallback Pairing（覆盖率不足）**
- 若 `misaligned 覆盖率 < DSAG_GAPLINK_MIN_COVERAGE`（默认 0.8）
  - 用 embedding 找相关候选
  - 过滤/打分保留“相关但错位”的 pair
  - 作为 `is_aligned=false` 的兜底对齐

### 3) Embedding（`dsag/embedding_index.py`）

只嵌入 **叶子节点**：
- 文本 = `label + description + aliases`
- 分别构建 expert/researcher leaf embedding
用途：
1) runtime 实时定位
2) fallback pairing

### 4) Runtime（`dsag/runtime.py`）

**定位**：embedding 匹配专家回答 / 研究者问题  
**Assistance 输出（仅三类）**：
- `bridge_templates`
- `checkout_template`
- `followup_questions`

**Follow‑up 规则**：
- 若 expert leaf 有 `attributes` → DeepDive（纵向深挖）
- 否则 → ExpandScope（基于 siblings 横向扩展）

**Polish（低延迟润色）**：
- 使用 `OPENAI_MODEL_POLISH`（默认 `qwen-turbo`）
- 输入：专家原话 + 当前问答 + 最近三轮 context 摘要 + 三类 assistance
- 输出：覆盖 `assistance` 文案（更贴合对话语境）

**Context 摘要来源（服务端）**
- `app.py` 从 session 取最近三轮 QA，规则截断生成摘要

---

## DSAG API（后端）

### 初始化 DSAG
`POST /api/dsag/init`

```json
{
  "topic": "...",
  "researcher_bg": "...",
  "expert_bg": "..."
}
```

### DSAG 状态
`GET /api/dsag/status`

### 分析一轮对话
`POST /api/dsag/analyze_turn`

```json
{
  "researcher_question": "...",
  "expert_answer": "..."
}
```

返回结构（重点）：
```json
{
  "analysis": {
    "located": {...},
    "divergence": {...},
    "assistance": {
      "bridge_templates": {...},
      "checkout_template": "...",
      "followup_questions": [...]
    }
  }
}
```

### 获取完整图谱
`GET /api/dsag/graph`

---

## 前端（static + templates）

前端实现位于：
- `templates/index.html`
- `static/app.js`
- `static/style.css`

主要功能：
- 访谈消息显示、术语高亮、按钮交互  
- Follow‑ups 可点击填入研究者输入框  
- DSAG 结果展示（由后端 API 提供）

---

## Questionnaire

将问卷放在项目根目录：
```
questionnaire.docx
```

系统会在构建 Researcher Tree 时自动读取该问卷内容（仅对 Agent B 可见）。若缺失则回退为原有生成逻辑。

---

## 常见问题

### 1) LLM 输出 JSON 失败
- 日志提示 “no perspectives returned”
- 解决：简化 prompt 或加入重试/调试输出

### 2) Embedding 报错（404/405）
- 检查 `OPENAI_EBD_API_KEY` / `OPENAI_EBD_BASE_URL`
- 确认模型名是否支持（如 `text-embedding-3-small`）

### 3) GapLink 数量过少
- 调整 `DSAG_GAPLINK_MIN_COVERAGE`
- 调整 fallback 参数（top‑K / sim 窗口 / alpha/beta）

---

## 说明

`llm_backend.py` 是早期 demo，可忽略；DSAG 核心逻辑全部在 `dsag/` 内。
