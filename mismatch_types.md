# Mismatch Types in Expert-Researcher Communication

### 1. Lexical Gap
- **Description**: Mismatches in terminology where parties use different words for the same concept, or jargon/abbreviations that obscure meaning.
- **Solution**: Translate/explain terms.
- **Implementation**: Perform a strict 1-to-1 semantic substitution. Map the term to a native vocabulary word (e.g., 'EHR' -> 'Database').

---

### 2. Conceptual Gap
- **Description**: The lack of a shared Mental Model or domain Common Sense causes fundamental communication barriers, leading to misinterpretations of underlying concepts and structural logic.
- **Solution** (No order):
  1. Construct analogies using familiar concepts from the target domain.
  2. Construct specific fine-grained scenarios (a highly specific, low-level concrete example).
- **Implementation**:
  - **Analogy Construction**: Do not merely swap nouns. Identify a 'Sibling Concept' in the expert's domain and map the relational structure (Inputs, Logic, Outputs). Explain how the researcher's concept behaves similarly to the expert's familiar concept.
  - **Scenario Rule**: Ensure the Fine-grained Scenario explicitly specifies inputs, outputs, and edge cases to prevent ambiguity. Avoid overarching metaphors that require secondary interpretation.

---

### 3. Tacit Gap
- **Description**: A disconnect between the expert's reliance on implicit, experience-based intuition and the researcher's need for explicit, quantifiable design parameters.
- **Solution** (There is a sequence):
  1. Extract attributes from expert answers.
  2. Guide experts to dismantle factors through quantifiable indicators.
  3. Come up with some hypothetical scenarios for checkout (allow user to decide whether to generate).
- **Implementation**:
  - **Probing Rule**: Never ask open-ended "Why" questions. Instead, formulate a multiple-choice probe using known domain attributes (e.g., "Is your concern related to [Attribute A] or [Attribute B]?").
  - **Checkout Design**: Alter only one variable in the hypothetical scenario at a time to test the boundaries of their intuition.

---

### 4. Scope Gap
- **Description**: The focus is inconsistent (Experts focus on practical functions, while researchers focus on research value) or there is a difference in their expectations regarding "what to do."
- **Solution** (There is a sequence):
  1. Validate the expert's practical focus.
  2. Inject the research goal by highlighting a potential limitation in their approach, using a Coarse-grained Scenario (a high-level, macro narrative that deliberately hides technical constraints) to persuade the expert of the research's value.
- **Implementation**:
  - **Injection Strategy**: Use a "Validate & Pivot" template (e.g., "Your focus on [Practical Utility] is highly applicable, but we might encounter [Potential Limitation]. Solving [Research Goal] helps us overcome this because...").
  - **Persuasion Rule**: Do not merely state the research is a "necessary step." Actively generate compelling reasons showing how the research safeguards or enhances the expert's interests.
  - **Scenario Rule**: Deploy the Coarse-grained Scenario specifically as a persuasive tool to visualize this macro-level benefit, bypassing granular technical flaws or budget arguments.

---

### 5. Process Gap
- **Description**: A lack of standardized procedures or narrow expert narratives disrupts the discussion process.
- **Solution** (Entirely runtime-driven — no offline pre-computation):
  1. **Monitoring Layer**: Maintain a real-time timeline that accumulates the actual step→step conversation history. Use this timeline together with the expert tree structure to detect three types of drift: repeated topics, tunnel vision, and coverage gaps.
  2. **Redirect Layer**: When drift is detected, make a runtime LLM call with full conversation context (timeline + expert's actual words + tree structure) to generate a natural, context-aware redirect — a sentence the researcher can directly speak.
- **Key Design Principle**: Process Gap is fundamentally different from the other 4 types. The other types address *static* structural mismatches that can be pre-computed offline. Process Gap addresses *dynamic* conversational flow problems that only materialize during the interview. Therefore, it uses no offline scaffold or pre-generated templates — all assistance is generated at runtime from real data.
- **Implementation**:
  - **Offline**: Agent C classifies the GapLink as ProcessGap. No dedicated prompt, no scaffold, no pre-generated payload. `assistance_payload` remains empty.
  - **Runtime Data Sources** (all grounded, no LLM hypotheses):
    1. `interview_timeline` — the real step→step conversation history, accumulated each turn.
    2. Expert tree structure — sibling leaves (related topics under the same L2 category) and parent categories provide the "topic universe" for coverage analysis.
    3. Current turn's `expert_answer` and `researcher_question` — allows the LLM to reference the expert's actual words in the redirect.
  - **Drift Detection** (rule-based, no LLM involved):
    1. **Repeated Topic**: Current `topic_label` already exists in timeline history. No external reference needed — purely based on actual conversation data.
    2. **Tunnel Vision**: The same `topic_label` or the same L2 category dominates the last N turns (threshold: ≥3 consecutive or near-consecutive turns on the same leaf/L2), while sibling leaves remain unvisited.
    3. **Coverage Gap**: Sibling leaves of the current expert leaf that have never appeared in the timeline. This uses the expert tree as the reference — a real, grounded structure, not an LLM-generated scaffold.
  - **Redirect Generation** (runtime LLM call, only when drift is detected):
    - Input: drift type + timeline summary + current expert answer + unvisited sibling labels + misalignment reason.
    - Output: one natural redirect sentence that references the expert's actual words and transitions to an unvisited topic.
    - When no drift is detected: return only the coverage map (structured data, no LLM call needed).
  - **Advisory Principle**: All alerts are suggestions, not directives. The researcher may dismiss any alert.
