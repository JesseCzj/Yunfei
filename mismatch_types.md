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
- **Description**: Factual errors disrupt the discussion process, as do a lack of standardized procedures or narrow expert narratives.
- **Solution**: Provide a real-time timeline or flowchart based on the ongoing interview to map the conversation, and issue alerts when the expert drifts off-topic or falls into narrative tunnel vision.
- **Implementation**:
  - **Presentation Form**: Dynamically generate and display a visual timeline/flowchart that maps the covered topics based on the current interview transcript.
