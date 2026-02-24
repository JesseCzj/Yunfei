# CONTEXT FOR CLAUDE CODE — Interview Assistant Improvement Task

---

## Part 1: Task Background & Design Decisions (From Researcher Interview)

### Overall Task Scope
- **Goal**: Modify the existing system's prompt design and frontend/backend to generate `assistance` corresponding to each of the 5 mismatch types.
- **What stays unchanged**: The graph/tree generation logic and the edges between nodes remain untouched.
- **What needs to change**: The `assistance` generation logic — specifically the prompts and node attribute design for each mismatch type.
- **Node attributes**: May need to be added or modified to support new assistance strategies.

### Key Design Decisions
1. **The assistance output is not limited to the original 3 fixed items** (Bridge Strategy / Checkout Strategy / Checkout Template). Each mismatch type should drive its own set of assistance items based on its solution needs — the number and type of assistance outputs can differ across mismatch types. The coarse/balanced/fine granularity within bridge templates can be retained where applicable.
2. **Each mismatch type can have a different number of bullet points/rules** — no need to force alignment across the 5 types.
3. **Use detailed, structured prompt descriptions** (similar to the verbose format with named strategies and rules), rather than minimalist 3-line templates — the complexity of each mismatch's solution requires sufficient context for the model to execute correctly.
4. **Some solutions have a strict sequence** (e.g., Tacit Gap, Scope Gap) — the prompt must enforce step order.
5. **Some solutions have no required order** (e.g., Conceptual Gap) — the prompt should treat strategies as parallel options.

---

## Part 2: Mismatch Type Definitions & Implementation Rules

### 1. Lexical Gap
- **Description**: Mismatches in terminology where parties use different words for the same concept, or jargon/abbreviations that obscure meaning.
- **Solution**: Translate/explain terms.
- **Implementation**: Perform a strict 1-to-1 semantic substitution. Map the term to a native vocabulary word (e.g., 'EHR' → 'Database').

---

### 2. Conceptual Gap
- **Description**: The lack of a shared Mental Model or domain Common Sense causes fundamental communication barriers, leading to misinterpretations of underlying concepts and structural logic.
- **Solution** (No required order — treat as parallel strategies):
  1. Construct analogies using familiar concepts from the target domain.
  2. Construct specific fine-grained scenarios (a highly specific, low-level concrete example).
- **Implementation**:
  - **Analogy Construction**: Do not merely swap nouns. Identify a 'Sibling Concept' in the expert's domain and map the relational structure (Inputs, Logic, Outputs). Explain how the researcher's concept behaves similarly to the expert's familiar concept.
  - **Scenario Rule**: Ensure the Fine-grained Scenario explicitly specifies inputs, outputs, and edge cases to prevent ambiguity. Avoid overarching metaphors that require secondary interpretation.

---

### 3. Tacit Gap
- **Description**: A disconnect between the expert's reliance on implicit, experience-based intuition and the researcher's need for explicit, quantifiable design parameters.
- **Solution** (Strict sequence — must follow this order):
  1. Extract attributes from expert answers.
  2. Guide experts to dismantle factors through quantifiable indicators.
  3. Come up with some hypothetical scenarios for checkout (allow user to decide whether to generate).
- **Implementation**:
  - **Probing Rule**: Never ask open-ended "Why" questions. Instead, formulate a multiple-choice probe using known domain attributes (e.g., "Is your concern related to [Attribute A] or [Attribute B]?").
  - **Checkout Design**: Alter only one variable in the hypothetical scenario at a time to test the boundaries of their intuition.

---

### 4. Scope Gap
- **Description**: The focus is inconsistent (Experts focus on practical functions, while researchers focus on research value) or there is a difference in their expectations regarding "what to do."
- **Solution** (Strict sequence — must follow this order):
  1. Validate the expert's practical focus.
  2. Inject the research goal by highlighting a potential limitation in their approach, using a Coarse-grained Scenario (a high-level, macro narrative that deliberately hides technical constraints) to persuade the expert of the research's value.
- **Implementation**:
  - **Injection Strategy**: Use a "Validate & Pivot" template (e.g., "Your focus on [Practical Utility] is highly applicable, but we might encounter [Potential Limitation]. Solving [Research Goal] helps us overcome this because...").
  - **Persuasion Rule**: Do not merely state the research is a "necessary step." Actively generate compelling reasons showing how the research safeguards or enhances the expert's interests.
  - **Scenario Rule**: Deploy the Coarse-grained Scenario specifically as a persuasive tool to visualize this macro-level benefit, bypassing granular technical flaws or budget arguments.

---

### 5. Process Gap
- **Description**: Factual errors disrupt the discussion process, as do a lack of standardized procedures or narrow expert narratives.
- **Solution**: Provide a real-time timeline or flowchart based on the ongoing interview to map the conversation, and issue alerts when the expert drifts off-topic or falls into narrative tunnel vision.
- **Implementation**:
  - **Presentation Form**: Dynamically generate and display a visual timeline/flowchart that maps the covered topics based on the current interview transcript.

---

## Summary: What Claude Code Should Focus On

1. **Modify the assistance generation prompts in `dsag/runtime.py` (and related factory prompts in `dsag/factory.py`)** so that each of the 5 `RelationType` gap types generates assistance tailored to its specific strategy and rules (as defined in Part 2 above).
2. **The assistance output structure is not fixed at 3 items** — do not assume every mismatch type must produce `bridge_templates` + `checkout_template` + `followup_questions`. Each mismatch type should produce the assistance items that its solution strategy actually requires. The coarse/balanced/fine granularity within bridge templates can be retained where it still applies.
3. **Add or modify node attributes as needed** — for example, Tacit Gap requires `attributes` on leaf nodes; other types may need additional fields.
4. **Do not change** the graph/tree construction logic or edge generation between nodes.
5. **After modifying prompts and schema**, connect frontend and backend and run integration tests.