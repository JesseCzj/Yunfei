# Test 1 Report

## Purpose
This report records the first structured manual test round for the new `Process Guidance` panel. It compares:

- the intended test design
- the predicted gap behavior before running
- the actual outputs returned by the system
- the resulting interpretation of system behavior

## Test Setup
- Topic: `AI-assisted Medical Diagnosis`
- Researcher background: `HCI researcher studying explainable AI`
- Expert background: `Radiologist with 10 years experience`
- UI mode: manual paired entry
  - first submit researcher question
  - then submit expert response
- App mode used during stable testing:

```powershell
python -m dotenv run -- python -c "from app import app; app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)"
```

## Important Context For Interpretation
- A visible DSAG card appears only when a non-empty assistance payload is returned.
- Some attempted turns were likely gated by low confidence and therefore did not show a visible assistance card.
- The process panel currently mixes two ideas:
  - session-level visualization from timeline state
  - process alert state still dependent on a message classified as `ProcessGap`

## Turn-By-Turn Input Log

### T1
`Researcher`  
When the AI flags a chest CT as abnormal, what do you usually examine first before deciding whether to trust it?

`Expert`  
I first check whether the case seems in-distribution, then I look at the lesion candidate, the malignancy score, and whether the confidence stays stable across adjacent slices.

### T2
`Researcher`  
So the heatmap is basically the model's reasoning process, right?

`Expert`  
Not really. The heatmap is only a visual cue. It does not reveal the whole reasoning process, and it does not guarantee the probability estimate is reliable.

### T3
`Researcher`  
When you say the probability estimate is reliable, do you mean the model is simply more accurate on easy cases?

`Expert`  
No. I mean calibration. A model can rank cases well but still assign probabilities that do not match real-world likelihood.

### T4
`Researcher`  
If we just change the heatmap color and make the interface cleaner, would that solve most of your trust concerns?

`Expert`  
Not really. My bigger concern is where the AI fits into the workflow, when I should override it, and how it affects responsibility during reporting.

### T5
`Researcher`  
In an ambiguous case, what tacit signals make you hesitate before accepting the AI suggestion?

`Expert`  
I look for whether the finding is anatomically plausible, whether the confidence matches the image quality, and whether the recommendation fits the patient's broader context.

### T6
`Researcher`  
Let us stay specifically on confidence calibration. If the AI score is 0.62 in an ambiguous case, what exactly goes through your mind?

`Expert`  
I ask whether that score is calibrated for this subgroup, whether neighboring CT slices show the same lesion pattern, whether image quality is lowering confidence, and whether the case is close to the clinical decision threshold.

### T7
`Researcher`  
If the score drops from 0.62 to 0.58, how do you interpret that change when the CT slices still show the same lesion pattern?

`Expert`  
I still look at calibration, but I also check whether the neighboring CT slices support the same lesion pattern, whether image quality is reducing confidence, and whether 0.58 is close enough to the clinical decision threshold to change my judgment.

### T8
`Researcher`  
If it drops again to 0.55, do you still reason about it through calibration and threshold sensitivity in the same way?

`Expert`  
Yes. I compare whether the score is still calibrated for this subgroup, whether the lesion pattern is consistent across adjacent CT slices, whether image quality is affecting the score, and whether 0.55 is still near the operating threshold.

### T9
`Researcher`  
If the score falls to 0.52 but the lesion still looks similar across slices, what changes in your reasoning?

`Expert`  
I still would not trust the number alone. I check calibration for this subgroup, whether the lesion appearance stays consistent across neighboring CT slices, whether image quality explains the lower score, and whether 0.52 crosses the clinical decision threshold.

### T10-T11
The later repeated-topic turns were reported in the session output as continued calibration-focused repetitions that still mapped to `Slice thickness and spacing sensitivity`, but the exact user-entered wording for those final repeated turns was not fully preserved in the chat log.

## Comparison Table

| Turn / Phase | Research intention | Predicted signal before test | Actual gap type | Actual divergence / focus | Process panel observation | Assessment |
| --- | --- | --- | --- | --- | --- | --- |
| T1 | Open trust-framing question about abnormal chest CT | `LexicalGap` or `ConceptualGap` | `ConceptualGap` | Divergence at `root`; focus matched to `Slice thickness and spacing sensitivity` | Panel activated successfully | Good end-to-end validation, but topic grounding was narrower and more technical than expected. |
| T2 | Clarify whether heatmap equals reasoning | `ConceptualGap` | `ConceptualGap` | Divergence at `L2`; focus matched to `Black-box predictions without visual justification` | Topic trail updated normally | Strong match. This was one of the most semantically convincing outputs in the round. |
| T3 | Clarify whether reliability means simple accuracy | `LexicalGap` or `ConceptualGap` | `TacitGap` | Focus matched to `Inconsistent reasoning across similar cases` | `Current Focus`, coverage, and trail all updated correctly | Plausible output. The system interpreted reliability judgment as an implicit expertise issue rather than a pure terminology clarification. |
| T4 | Shift discussion to workflow, override, and reporting responsibility | `ScopeGap` or `ConceptualGap` | `ConceptualGap` | Divergence at `root`; focus matched to `Inadequate documentation standards for AI use` | New branch visible in local coverage | Reasonable but somewhat drifted toward audit/documentation language instead of broader workflow placement. |
| T5 | Probe tacit signals in ambiguous cases | `TacitGap` | `ConceptualGap` | Focus matched to `Fear of being second-guessed by non-radiologist reviewers` | Trail continued updating; no process alert | Semantically off target. The model stayed on the prior institutional/accountability branch rather than returning to tacit clinical cues. |
| T6 first calibration re-anchor | Re-anchor discussion to confidence calibration | Stable visible analysis in the calibration branch | Mixed: at least one attempt likely gated by low confidence; later visible outputs were `ConceptualGap` | Calibration attempts eventually collapsed into `Slice thickness and spacing sensitivity` | User experience became ambiguous because low-confidence attempts produced no visible DSAG card | Important finding: low-confidence turns currently look too similar to “no analysis happened.” |
| T7 | Repeated calibration question with `0.58` score | `ProcessGap` should begin to accumulate conditions | `ConceptualGap` | Focus stayed on `Slice thickness and spacing sensitivity` | Repeated topic trail continued | Repetition began to become visible, but not promoted to process alerting. |
| T8 | Repeated calibration question with `0.55` score | Stronger chance of process drift | `ConceptualGap` | Focus stayed on `Slice thickness and spacing sensitivity` | Trail and local coverage remained in same branch | Stable repeated-topic behavior, still no visible process signal. |
| T9 | Repeated calibration question with `0.52` score | `ProcessGap` should be close to visible | `ConceptualGap` | Focus stayed on `Slice thickness and spacing sensitivity` | Continued fixation in the same branch | By this point the process-level repetition was already evident in panel behavior. |
| T10-T11 repeated-topic phase | Continue forcing the same calibration-centered topic | `ProcessGap` should eventually appear in the panel | Visible analyzed turns remained `ConceptualGap` | `Current Focus` repeatedly stayed `Slice thickness and spacing sensitivity` | `Recent Topic Trail` clearly showed repeated topic fixation through `T11`; `Latest Process Signal` stayed empty | This is the central finding of Test 1. Process-level repetition is visible, but process alert logic does not activate independently. |

## Consolidated Output Snapshot

### Stable panel behavior observed
- `Current Focus` updated when visible analyzed turns were produced.
- `Coverage View` correctly behaved as local branch coverage, not global session coverage.
- `Recent Topic Trail` accumulated repeated conceptual focus across later turns.

### Stable semantic behavior observed
- The system can output multiple gap types, not only one fixed category.
- `ConceptualGap` outputs often had coherent analogy/scenario structure.
- `TacitGap` can appear and be rendered correctly.

### Reproducible problems observed
- Runtime topic matching can collapse into overly specific technical leaves.
- Once the conversation drifts into a branch such as documentation/accountability, later turns may continue to inherit that framing.
- Low-confidence gating is not sufficiently exposed in the interface.
- The process panel can show repeated topic fixation without surfacing a process alert.

## Evaluation

### What worked well
- The new panel is useful and visually meaningful.
- The testing transcript was sufficient to expose process-level behavior.
- The app is now much better at showing session trajectory than before.

### What did not work as intended
- `Latest Process Signal` did not activate during a clear repeated-topic segment.
- The system therefore under-reported process drift even when the panel itself visually demonstrated it.

## Main Interpretation
The strongest conclusion from this test is:

**process visualization is working, but process alerting is still coupled too tightly to primary gap classification.**

Observed evidence:
- `T6-T11` repeatedly remained on the same focus
- local branch coverage stayed in the same region
- the trail made repetition obvious
- but the alert section remained empty

## Recommended Follow-Up
1. Decouple process drift detection in the right-side panel from `selected_link.relation_type == "ProcessGap"`.
2. Keep per-turn semantic gap classification for the message card unchanged.
3. Compute `repeated_topic` and `tunnel_vision` directly from timeline state for the panel.
4. Show low-confidence warnings even when no assistance payload is rendered.

## Bottom Line
Test 1 did not merely show a missing alert. It demonstrated a concrete architectural issue:

- the session-level process behavior is already being captured,
- but the current panel does not yet promote that behavior into an independent `ProcessGap` signal.
