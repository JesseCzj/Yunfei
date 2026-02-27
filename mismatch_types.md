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
- **Two-Phase Design** (offline exhaustive preparation → runtime intelligent filtering):
  - **Offline Phase (Payload Generation)**: When Agent C processes a TacitGap link, it should **expand the attribute list** beyond Agent A's initial 2-4 attributes. Given the mismatch information and the current related attributes, generate as many possible/potential related attributes as reasonable. Then for ALL attributes (Agent A's original + newly expanded), generate step 2 multi-select probes and step 3 hypothetical scenario checkouts. The goal is to build an **exhaustive arsenal** of probes and scenarios offline.
  - **Runtime Phase (Polish LLM)**: The polish LLM's role changes from simple rephrasing to **intelligent filtering**:
    1. **Extract** which attributes the expert has already mentioned/articulated during the live interview.
    2. **Filter out** the already-mentioned attributes from the full attribute list.
    3. **Present** the assist (probes + scenarios) only for the **remaining unmentioned attributes**.
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
  1. **Coverage Analysis**: Maintain a real-time timeline that accumulates the actual step→step conversation history. Combined with the expert tree's sibling structure, always compute and return coverage info (visited topics, unvisited siblings, coverage ratio) as informational data.
  2. **Drift Detection**: Use timeline + tree structure to detect two types of drift alert: Repeated Topic and Tunnel Vision.
  3. **Redirect Generation**: When drift is detected, make a runtime LLM call with full conversation context to generate a natural, context-aware redirect sentence the researcher can directly speak.
- **Key Design Principle**: Process Gap is fundamentally different from the other 4 types. The other types address *static* structural mismatches that can be pre-computed offline. Process Gap addresses *dynamic* conversational flow problems that only materialize during the interview. Therefore, it uses no offline scaffold or pre-generated templates — all assistance is generated at runtime from real data.
- **Implementation**:
  - **Offline**: Agent C classifies the GapLink as ProcessGap. No dedicated prompt, no LLM call. `assistance_payload` stores only `{"misalignment_reason": reason}` (Agent C's alignment reason, for runtime redirect prompt consumption).
  - **Runtime Data Sources** (all grounded, no LLM hypotheses):
    1. `interview_timeline` — the real step→step conversation history, accumulated each turn.
    2. Expert tree structure — sibling leaves (related topics under the same L2 category) provide the "topic universe" for coverage analysis.
    3. Current turn's `expert_answer` and `researcher_question` — allows the LLM to reference the expert's actual words in the redirect.
    4. `misalignment_reason` — Agent C's alignment reason stored in `assistance_payload`, injected into the redirect prompt.
  - **Drift Detection** (rule-based, no LLM, two alert types checked in priority order — mutually exclusive):
    1. **Repeated Topic** (priority 1): Current `topic_label` has appeared ≥2 times in timeline history (i.e., 3rd+ discussion of this topic). Purely based on actual conversation data.
    2. **Tunnel Vision** (priority 2): The last 4 turns all map to the same `expert_leaf_id`, AND there exist unvisited sibling leaves. The sibling condition prevents false positives — if all siblings are covered, deep-diving is reasonable.
  - **Coverage Gap** (info, not alert — always computed):
    - Unvisited sibling leaves of the current expert leaf, computed from timeline + tree structure.
    - Returned as structured data (`{visited, unvisited_siblings, coverage_ratio}`) in every ProcessGap response.
    - Does NOT trigger LLM redirect — it is informational, not a drift signal. The researcher decides whether to explore uncovered topics.
  - **Redirect Generation** (runtime LLM call, only when Repeated Topic or Tunnel Vision is detected):
    - Input: drift type + drift detail + timeline summary (last 4 turns) + current expert answer (truncated to 500 chars) + unvisited sibling labels + descriptions + misalignment reason.
    - Output: one natural redirect sentence that references the expert's actual words and transitions to an unvisited topic.
    - When no drift is detected: no LLM call, `redirect` is null. Coverage info still returned.
  - **Skip Polish**: ProcessGap skips the `_polish_assistance()` step — the redirect is already LLM-generated with full conversation context.
  - **Advisory Principle**: All alerts and redirects are suggestions, not directives. The researcher may dismiss any alert.
