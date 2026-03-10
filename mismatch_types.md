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
- **Description**: A mismatch that risks disrupting the interview process. Originates from three root causes: (1) factual errors in the researcher's preparation, (2) multiple paradigms in the domain that the researcher doesn't anticipate, (3) narrow expert narratives that prevent topic expansion. These are handled by a **two-layer defense system**.

#### Two-Layer Architecture

```
ProcessGap Defense
├── Prevention Layer (offline, Agent C) — handles #1 and #2
│    ├── #1 Factual Risk: marks factual vulnerabilities in the researcher's preparation
│    └── #2 Methodology Conflict: marks areas where multiple paradigms exist
│    → Output: structured ProcessGap assistance (parallel to other 4 gap types)
│
└── Monitoring Layer (runtime, every turn, independent of gap type) — handles #3
     └── Narrow Focus detection (single signal)
     → Output: DriftSignal (coverage + narrow focus alert + redirect)
```

The two layers are **orthogonal**: a single turn can produce both a ProcessGap assistance card AND a drift alert simultaneously.

#### Prevention Layer (Offline)
- **Agent C Classification**: Agent C identifies ProcessGap when comparing two leaf nodes reveals either:
  - *Factual Risk*: The researcher's preparation contains an assumption that may contradict established domain practice. If stated as fact, the expert will correct it, derailing the discussion.
  - *Methodology Conflict*: The domain has multiple valid approaches for a task, but the researcher only anticipates one. Confirmatory questions would miss the expert's actual practice.
- **Payload Generation**: A dedicated `PROCESS_GAP_PROMPT` generates structured assistance:
  - Factual Risk: `{sub_type, vulnerable_assumption, domain_correction, safe_phrasing, misalignment_reason}`
  - Methodology Conflict: `{sub_type, known_approaches, researcher_assumed_approach, open_process_question, misalignment_reason}`
- **Polish**: At runtime, the `safe_phrasing` or `open_process_question` is polished by the context-aware LLM to sound natural and reference the expert's recent answer.

#### Monitoring Layer (Runtime)
- **Runs every turn**, independent of gap type. A turn can trigger drift detection regardless of whether the matched link is ProcessGap, ConceptualGap, or any other type.
- **Data Sources** (all grounded, no LLM hypotheses):
  1. `interview_timeline` — accumulated step→step conversation history.
  2. Expert tree structure — sibling leaves provide the "topic universe" for coverage analysis.
  3. Current turn's Q&A text — for LLM redirect generation.
- **Coverage** (informational, always computed): `{visited, unvisited_siblings, coverage_ratio}`.
- **Narrow Focus Detection** (rule-based, no LLM, single signal): Fires when the recent conversation window (last 4-6 turns) has low topic diversity (distinct topics < half the window size) AND unvisited sibling topics exist. The detail text dynamically describes the specific pattern (how many turns, which topics, how many unexplored).
- **Redirect Generation** (runtime LLM call, only when drift detected): Produces one natural redirect sentence referencing the expert's actual words.
- **Advisory Principle**: All alerts and redirects are suggestions, not directives.
