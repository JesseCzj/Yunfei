#!/usr/bin/env python3
"""
DSAG v2 Smoke Test Script

Tests the updated DSAG backend with type-specific assistance generation.
Run with: python test_dsag_v2.py

Tests:
1. Schema: GapLink with assistance_payload, DSAGState with interview_timeline
2. Factory: Tree building (no API), type-specific prompt dispatch structure
3. Runtime: Type-specific assistance branching, timeline accumulation, drift detection
4. Full generation (optional, requires API key)
"""

import os
import sys
import json
from datetime import datetime

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed, using existing environment variables")


def test_schema():
    """Test updated DSAG schema classes."""
    print("\n=== Testing Schema (v2) ===")

    from dsag.schema import (
        DSAGNode, TaxonomyTree, GapLink, DSAGGraph, DSAGState,
        Tower, Layer, RelationType, validate_graph, compute_lca_layer
    )

    # Test node creation
    node = DSAGNode(
        id="test_node",
        tower=Tower.EXPERT.value,
        layer=Layer.LEAF.value,
        label="Test Pain Point",
        description="A test pain point",
        parent_id="test_parent",
        attributes=["confidence level", "noise threshold"],
    )
    assert node.id == "test_node"
    assert node.is_leaf()
    assert len(node.attributes) == 2
    print("  [OK] Node creation with attributes")

    # Test LCA computation
    path1 = ["Root", "Perspective A", "Category 1", "Leaf 1"]
    path2 = ["Root", "Perspective A", "Category 2", "Leaf 2"]
    lca = compute_lca_layer(path1, path2)
    assert lca == Layer.L1.value, f"Expected L1, got {lca}"
    print("  [OK] LCA computation")

    # Test GapLink with assistance_payload (no more BridgeTemplates)
    link = GapLink(
        expert_leaf_id="exp_leaf_00_00_00",
        researcher_leaf_id="res_leaf_00_00_00",
        relation_type=RelationType.LEXICAL_GAP.value,
        assistance_payload={
            "term_mapping": {
                "expert_term": "EHR",
                "researcher_term": "Database",
                "explanation": "EHR is the expert's term for electronic health records.",
            }
        },
        weight=7.5,
    )
    assert link.assistance_payload["term_mapping"]["expert_term"] == "EHR"
    assert not hasattr(link, "bridge_templates"), "bridge_templates should not exist"
    assert not hasattr(link, "checkout_template"), "checkout_template should not exist"
    print("  [OK] GapLink with assistance_payload (LexicalGap)")

    # Test GapLink serialization round-trip
    link_dict = link.to_dict()
    assert "assistance_payload" in link_dict
    assert "bridge_templates" not in link_dict
    link_restored = GapLink.from_dict(link_dict)
    assert link_restored.assistance_payload == link.assistance_payload
    assert link_restored.relation_type == RelationType.LEXICAL_GAP.value
    print("  [OK] GapLink serialization round-trip")

    # Test different payload shapes per type
    tacit_link = GapLink(
        expert_leaf_id="exp_leaf_00_00_01",
        researcher_leaf_id="res_leaf_00_00_00",
        relation_type=RelationType.TACIT_GAP.value,
        assistance_payload={
            "attributes": ["confidence level", "noise threshold"],
            "probes": [
                {
                    "attribute": "confidence level",
                    "question": "Is your concern more about low confidence or inconsistent confidence?",
                    "choices": ["Low confidence", "Inconsistent confidence"],
                }
            ],
            "hypothetical_scenario": "If the confidence score were always above 90%, would you still rely on your intuition?",
        },
        weight=8.0,
    )
    assert len(tacit_link.assistance_payload["probes"]) == 1
    print("  [OK] GapLink with assistance_payload (TacitGap)")

    scope_link = GapLink(
        expert_leaf_id="exp_leaf_00_01_00",
        researcher_leaf_id="res_leaf_00_00_00",
        relation_type=RelationType.SCOPE_GAP.value,
        assistance_payload={
            "validate_focus": "Your focus on minimizing false alarms is crucial for clinical workflow.",
            "pivot": {
                "limitation": "Without understanding why alarms trigger, you can't distinguish true from false.",
                "research_goal": "Explainable alert reasoning",
                "compelling_reason": "This would let you dismiss false alarms in seconds instead of minutes.",
                "coarse_scenario": "Imagine a morning round where each alert shows its top 3 reasons. You glance, confirm, and move on.",
            },
        },
        weight=6.5,
    )
    assert "pivot" in scope_link.assistance_payload
    print("  [OK] GapLink with assistance_payload (ScopeGap)")

    # Test DSAGState with interview_timeline
    state = DSAGState()
    assert state.interview_timeline == []
    state.interview_timeline.append({
        "turn_index": 1,
        "topic_label": "False positives",
        "expert_leaf_id": "exp_leaf_00_00_00",
        "researcher_leaf_id": "res_leaf_00_00_00",
        "relation_type": "LexicalGap",
        "summary": "Q: How do you handle alerts? | A: Too many false alarms.",
    })
    assert len(state.interview_timeline) == 1
    print("  [OK] DSAGState with interview_timeline")

    print("  Schema v2 tests passed!")
    return True


def test_graph_factory_mock():
    """Test GraphFactory tree building and link construction (no API calls)."""
    print("\n=== Testing GraphFactory (v2 structure) ===")

    from dsag.factory import (
        _build_expert_tree, _build_researcher_tree,
        build_links_from_alignments,
    )
    from dsag.schema import (
        TaxonomyTree, NodeAlignment, TreeAlignments,
        RelationType, Layer,
    )

    # Mock LLM response for expert
    mock_expert_response = {
        "perspectives": [
            {
                "label": "Clinical Outcomes",
                "description": "Concerns about patient results",
                "categories": [
                    {
                        "label": "Diagnosis Accuracy",
                        "description": "Getting the right diagnosis",
                        "pain_points": [
                            {
                                "label": "False positives",
                                "description": "Too many incorrect alerts",
                                "is_intuition": False,
                            },
                            {
                                "label": "Uncertainty in edge cases",
                                "description": "Hard to decide on borderline cases",
                                "is_intuition": True,
                                "attributes": ["confidence level", "patient history completeness"],
                            },
                        ]
                    },
                    {
                        "label": "Alert Fatigue",
                        "description": "Too many notifications",
                        "pain_points": [
                            {
                                "label": "Alarm overload",
                                "description": "Constant beeping leads to desensitization",
                                "is_intuition": False,
                            },
                        ]
                    }
                ]
            }
        ]
    }

    mock_researcher_response = {
        "perspectives": [
            {
                "label": "Model Interpretability",
                "description": "Understanding AI decisions",
                "categories": [
                    {
                        "label": "Feature Importance",
                        "description": "Which inputs matter most",
                        "goals": [
                            {
                                "label": "Visualize feature contributions",
                                "description": "Show which features led to prediction",
                                "hci_terms": ["SHAP values", "feature attribution"],
                            },
                            {
                                "label": "Confidence calibration display",
                                "description": "Show how reliable the model's confidence is",
                                "hci_terms": ["calibration plot", "reliability diagram"],
                            },
                        ]
                    }
                ]
            }
        ]
    }

    topic = "AI-assisted Medical Diagnosis"
    expert_tree = _build_expert_tree(mock_expert_response, topic)
    researcher_tree = _build_researcher_tree(mock_researcher_response, topic)

    assert len(expert_tree.get_leaves()) == 3
    assert len(researcher_tree.get_leaves()) == 2
    print(f"  [OK] Expert tree: {len(expert_tree.nodes)} nodes, {len(expert_tree.get_leaves())} leaves")
    print(f"  [OK] Researcher tree: {len(researcher_tree.nodes)} nodes, {len(researcher_tree.get_leaves())} leaves")

    # Check attributes preserved on intuition nodes
    intuition_nodes = [n for n in expert_tree.get_leaves() if n.attributes]
    assert len(intuition_nodes) == 1
    print(f"  [OK] Intuition node has attributes: {intuition_nodes[0].attributes}")

    # Test link construction from mock alignments
    alignments = TreeAlignments(
        leaf_alignments=[
            NodeAlignment(
                expert_node_id="exp_leaf_00_00_00",
                researcher_node_id="res_leaf_00_00_00",
                is_aligned=False,
                reason="Different terminology: expert uses clinical alert terms, researcher uses ML terms",
                semantic_similarity=0.3,
            ),
            NodeAlignment(
                expert_node_id="exp_leaf_00_00_01",
                researcher_node_id="res_leaf_00_00_01",
                is_aligned=False,
                reason="Expert relies on intuition and tacit criteria, researcher wants quantifiable metrics",
                semantic_similarity=0.4,
            ),
            NodeAlignment(
                expert_node_id="exp_leaf_00_01_00",
                researcher_node_id="res_leaf_00_00_00",
                is_aligned=False,
                reason="Expert focuses on broad disruption scope, researcher has a narrower goal and different focus boundary",
                semantic_similarity=0.25,
            ),
        ],
        concept_alignments=[
            NodeAlignment(
                expert_node_id="shared_root",
                researcher_node_id="shared_root",
                is_aligned=True,
                reason="Same topic",
                semantic_similarity=1.0,
            ),
        ],
    )

    links = build_links_from_alignments(expert_tree, researcher_tree, alignments)
    assert len(links) == 3, f"Expected 3 links, got {len(links)}"

    # Verify links have empty assistance_payload (to be filled by LLM)
    for link in links:
        assert link.assistance_payload == {}, f"Expected empty payload, got {link.assistance_payload}"
        assert link.relation_type in [r.value for r in RelationType]

    # Check inferred relation types from reason text
    relation_types = {link.expert_leaf_id: link.relation_type for link in links}
    assert relation_types["exp_leaf_00_00_00"] == RelationType.LEXICAL_GAP.value, \
        f"Expected LexicalGap, got {relation_types['exp_leaf_00_00_00']}"
    assert relation_types["exp_leaf_00_00_01"] == RelationType.TACIT_GAP.value, \
        f"Expected TacitGap, got {relation_types['exp_leaf_00_00_01']}"
    assert relation_types["exp_leaf_00_01_00"] == RelationType.SCOPE_GAP.value, \
        f"Expected ScopeGap, got {relation_types['exp_leaf_00_01_00']}"
    print("  [OK] Relation types correctly inferred from reason text")

    print("  GraphFactory v2 tests passed!")
    return True


def test_runtime_type_branching():
    """Test RuntimeEngine type-specific assistance generation (no API calls)."""
    print("\n=== Testing Runtime Type Branching ===")

    from dsag.schema import (
        DSAGNode, TaxonomyTree, GapLink, DSAGGraph,
        Tower, Layer, RelationType,
    )
    from dsag.runtime import RuntimeEngine, Assistance

    # Build minimal trees for testing
    topic = "AI-assisted Medical Diagnosis"

    # Expert tree
    expert_nodes = {
        "shared_root": DSAGNode(id="shared_root", tower=Tower.SHARED.value, layer=Layer.ROOT.value,
                                label=topic, parent_id=None, children_ids=["exp_L1_00"]),
        "exp_L1_00": DSAGNode(id="exp_L1_00", tower=Tower.EXPERT.value, layer=Layer.L1.value,
                              label="Clinical Outcomes", parent_id="shared_root", children_ids=["exp_L2_00_00"]),
        "exp_L2_00_00": DSAGNode(id="exp_L2_00_00", tower=Tower.EXPERT.value, layer=Layer.L2.value,
                                 label="Diagnosis Accuracy", parent_id="exp_L1_00",
                                 children_ids=["exp_leaf_00_00_00", "exp_leaf_00_00_01", "exp_leaf_00_00_02"]),
        "exp_leaf_00_00_00": DSAGNode(id="exp_leaf_00_00_00", tower=Tower.EXPERT.value, layer=Layer.LEAF.value,
                                      label="False positives", description="Too many incorrect alerts",
                                      parent_id="exp_L2_00_00"),
        "exp_leaf_00_00_01": DSAGNode(id="exp_leaf_00_00_01", tower=Tower.EXPERT.value, layer=Layer.LEAF.value,
                                      label="Uncertainty in edge cases", description="Hard to decide on borderline",
                                      parent_id="exp_L2_00_00",
                                      attributes=["confidence level", "patient history completeness"]),
        "exp_leaf_00_00_02": DSAGNode(id="exp_leaf_00_00_02", tower=Tower.EXPERT.value, layer=Layer.LEAF.value,
                                      label="Alarm overload", description="Constant beeping",
                                      parent_id="exp_L2_00_00"),
    }
    expert_tree = TaxonomyTree(tower=Tower.EXPERT.value, root_id="shared_root", nodes=expert_nodes)

    # Researcher tree
    researcher_nodes = {
        "shared_root": DSAGNode(id="shared_root", tower=Tower.SHARED.value, layer=Layer.ROOT.value,
                                label=topic, parent_id=None, children_ids=["res_L1_00"]),
        "res_L1_00": DSAGNode(id="res_L1_00", tower=Tower.RESEARCHER.value, layer=Layer.L1.value,
                              label="Model Interpretability", parent_id="shared_root", children_ids=["res_L2_00_00"]),
        "res_L2_00_00": DSAGNode(id="res_L2_00_00", tower=Tower.RESEARCHER.value, layer=Layer.L2.value,
                                 label="Feature Importance", parent_id="res_L1_00",
                                 children_ids=["res_leaf_00_00_00"]),
        "res_leaf_00_00_00": DSAGNode(id="res_leaf_00_00_00", tower=Tower.RESEARCHER.value, layer=Layer.LEAF.value,
                                      label="Visualize feature contributions",
                                      description="Show which features led to prediction",
                                      parent_id="res_L2_00_00",
                                      aliases=["SHAP values", "feature attribution"]),
    }
    researcher_tree = TaxonomyTree(tower=Tower.RESEARCHER.value, root_id="shared_root", nodes=researcher_nodes)

    # Create links with different types and pre-populated payloads
    links = [
        # LexicalGap link
        GapLink(
            expert_leaf_id="exp_leaf_00_00_00",
            researcher_leaf_id="res_leaf_00_00_00",
            relation_type=RelationType.LEXICAL_GAP.value,
            assistance_payload={
                "term_mapping": {
                    "expert_term": "false positive",
                    "researcher_term": "Type I error",
                    "explanation": "Both refer to incorrectly flagging a negative case as positive.",
                }
            },
            weight=7.0,
        ),
        # TacitGap link
        GapLink(
            expert_leaf_id="exp_leaf_00_00_01",
            researcher_leaf_id="res_leaf_00_00_00",
            relation_type=RelationType.TACIT_GAP.value,
            assistance_payload={
                "attributes": ["confidence level"],
                "probes": [
                    {
                        "attribute": "confidence level",
                        "question": "Is it more about low confidence or inconsistent confidence?",
                        "choices": ["Low confidence", "Inconsistent confidence"],
                    }
                ],
                "hypothetical_scenario": "If confidence were always >90%, would you still override?",
            },
            weight=8.5,
        ),
        # ConceptualGap link
        GapLink(
            expert_leaf_id="exp_leaf_00_00_02",
            researcher_leaf_id="res_leaf_00_00_00",
            relation_type=RelationType.CONCEPTUAL_GAP.value,
            assistance_payload={
                "analogy": {
                    "source_concept": "Triage nurse filtering patients",
                    "structural_mapping": {
                        "inputs": "Incoming alert signals",
                        "logic": "Priority scoring and filtering",
                        "outputs": "Actionable vs dismissible alerts",
                    },
                },
                "scenario": {
                    "inputs": "50 alerts in one shift",
                    "outputs": "5 marked critical, 45 auto-dismissed with reasons",
                    "edge_cases": "An alert dismissed by the system that the doctor would have flagged",
                },
            },
            weight=6.0,
        ),
    ]

    graph = DSAGGraph(
        topic=topic,
        researcher_bg="HCI researcher",
        expert_bg="Radiologist",
        expert_tree=expert_tree,
        researcher_tree=researcher_tree,
        links=links,
    )

    # We need a mock embedding index — create a minimal one
    class MockEmbeddingIndex:
        def search_expert_leaves(self, query, top_k=3):
            return []
        def search_researcher_leaves(self, query, top_k=3):
            return []

    engine = RuntimeEngine(graph, MockEmbeddingIndex())

    # Test 1: LexicalGap assistance
    assistance = engine.generate_assistance("exp_leaf_00_00_00", links[0])
    assert assistance.relation_type == RelationType.LEXICAL_GAP.value
    assert "term_mapping" in assistance.payload
    print("  [OK] LexicalGap: term_mapping present")

    # Test 2: TacitGap assistance
    assistance = engine.generate_assistance("exp_leaf_00_00_01", links[1])
    assert assistance.relation_type == RelationType.TACIT_GAP.value
    assert "probes" in assistance.payload
    assert "attributes" in assistance.payload
    assert len(assistance.payload["attributes"]) == 2, "Should merge node attributes"
    print(f"  [OK] TacitGap: {len(assistance.payload['probes'])} probes, "
          f"{len(assistance.payload['attributes'])} attributes")

    # Test 3: ConceptualGap assistance
    assistance = engine.generate_assistance("exp_leaf_00_00_02", links[2])
    assert assistance.relation_type == RelationType.CONCEPTUAL_GAP.value
    assert "analogy" in assistance.payload
    assert "scenario" in assistance.payload
    print("  [OK] ConceptualGap: analogy + scenario present")

    # Test 4: ProcessGap — offline payload only (drift is now independent)
    process_link = GapLink(
        expert_leaf_id="exp_leaf_00_00_00",
        researcher_leaf_id="res_leaf_00_00_00",
        relation_type=RelationType.PROCESS_GAP.value,
        assistance_payload={"misalignment_reason": "Expert lacks standardized workflow"},
        weight=5.0,
    )

    assistance = engine.generate_assistance("exp_leaf_00_00_00", process_link)
    assert assistance.relation_type == RelationType.PROCESS_GAP.value
    assert "misalignment_reason" in assistance.payload
    print(f"  [OK] ProcessGap: timeline={len(assistance.payload['timeline'])} entries, "
          f"drift_alerts={len(assistance.payload['drift_alerts'])}")

    # Test 5: Assistance serialization
    assist_dict = assistance.to_dict()
    assert "relation_type" in assist_dict
    assert "payload" in assist_dict
    assert "followup_questions" not in assist_dict, "followup_questions removed"
    assert "bridge_templates" not in assist_dict, "Old field should not exist"
    assert "checkout_template" not in assist_dict, "Old field should not exist"
    print("  [OK] Assistance.to_dict() has correct shape (no old fields)")

    print("  Runtime type branching tests passed!")
    return True


def test_tacit_postprocess_extracted_attributes():
    """TacitGap should display extracted attributes and remove their probes/scenarios."""
    print("\n=== Testing TacitGap Extracted Attributes Postprocess ===")

    from dsag.schema import RelationType
    from dsag.runtime import RuntimeEngine

    class DummyEngine(RuntimeEngine):
        pass

    # Bypass __init__ requirements; this test only calls a pure helper.
    engine = DummyEngine.__new__(DummyEngine)

    payload = {
        "attributes": ["confidence level", "patient history completeness", "signal quality"],
        "probes": [
            {
                "attribute": "confidence level",
                "question": "Is it low confidence or inconsistent confidence?",
                "choices": ["Low confidence", "Inconsistent confidence"],
            },
            {
                "attribute": "signal quality",
                "question": "Is signal quality mostly noisy or sparse?",
                "choices": ["Noisy", "Sparse"],
            },
        ],
        "hypothetical_scenarios": [
            "If confidence level rose from 60% to 95%, would your judgment change?",
            "If signal quality improved with less noise, would your judgment change?",
        ],
        "mentioned_attributes": ["confidence level"],
    }

    out = engine._postprocess_tacit_payload(payload)

    assert out["extracted_attributes"] == ["confidence level"]
    assert out["mentioned_attributes"] == ["confidence level"]
    assert out["attributes"] == ["patient history completeness", "signal quality"]
    assert len(out["probes"]) == 1
    assert out["probes"][0]["attribute"] == "signal quality"
    assert len(out["hypothetical_scenarios"]) == 1
    assert "confidence level" not in out["hypothetical_scenarios"][0].lower()
    print("  [OK] Extracted attributes are displayed and filtered from probes/scenarios")

    # Also verify alias input key is supported.
    payload2 = {
        "attributes": ["a", "b"],
        "probes": [],
        "hypothetical_scenarios": [],
        "extracted_attributes": ["a"],
    }
    out2 = engine._postprocess_tacit_payload(payload2)
    assert out2["attributes"] == ["b"]
    assert out2["mentioned_attributes"] == ["a"]
    print("  [OK] extracted_attributes alias is normalized")

    return True


def test_full_dsag_generation(skip_if_no_key=True):
    """Test full DSAG generation with real API calls."""
    print("\n=== Testing Full DSAG Generation (v2) ===")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        if skip_if_no_key:
            print("  [SKIP] No valid OPENAI_API_KEY, skipping full generation test")
            return True
        else:
            raise ValueError("OPENAI_API_KEY required for full test")

    from dsag import create_dsag_graph, build_embedding_index, RuntimeEngine

    topic = "AI-assisted Medical Image Analysis"
    researcher_bg = "HCI researcher focusing on explainable AI and visualization"
    expert_bg = "Radiologist with 10 years experience in medical imaging"

    print(f"  Generating DSAG for topic: {topic}")
    print("  This may take 30-60 seconds...")

    start = datetime.now()
    graph = create_dsag_graph(topic, researcher_bg, expert_bg, questionnaire="")
    duration = (datetime.now() - start).total_seconds()

    print(f"  [OK] Graph generated in {duration:.1f}s")
    print(f"       Expert leaves: {len(graph.expert_tree.get_leaves())}")
    print(f"       Researcher leaves: {len(graph.researcher_tree.get_leaves())}")
    print(f"       Links: {len(graph.links)}")

    # Verify links have assistance_payload (not bridge_templates)
    for link in graph.links[:3]:
        assert isinstance(link.assistance_payload, dict), "assistance_payload should be dict"
        assert not hasattr(link, "bridge_templates") or not isinstance(
            getattr(link, "bridge_templates", None), object
        ), "Should not have BridgeTemplates object"
        print(f"       Link ({link.relation_type}): payload keys = {list(link.assistance_payload.keys())}")

    # Test embeddings
    print("  Building embeddings index...")
    index = build_embedding_index(graph)
    print(f"  [OK] Embeddings built")

    # Test runtime
    print("  Testing runtime analysis...")
    engine = RuntimeEngine(graph, index)

    test_question = "How do you interpret the AI's confidence score?"
    test_answer = "I look at the highlighted areas, but honestly sometimes I just ignore it when it doesn't match my intuition."

    analysis = engine.analyze_turn(test_question, test_answer, context_summary="", interview_timeline=[])

    print(f"  [OK] Analysis completed")
    print(f"       Expert position: {analysis.located.best_expert_leaf_id} "
          f"(score={analysis.located.expert_confidence:.2f})")

    if analysis.assistance:
        print(f"       Assistance type: {analysis.assistance.relation_type}")
        print(f"       Payload keys: {list(analysis.assistance.payload.keys())}")
        print(f"       Payload keys: {list(analysis.assistance.payload.keys())}")

    print("  Full DSAG v2 generation test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("DSAG v2 Smoke Test (Type-Specific Assistance)")
    print("=" * 60)

    results = {}

    # Test 1: Schema
    try:
        results["schema_v2"] = test_schema()
    except Exception as e:
        print(f"  [FAIL] Schema test: {e}")
        import traceback; traceback.print_exc()
        results["schema_v2"] = False

    # Test 2: GraphFactory structure
    try:
        results["factory_v2"] = test_graph_factory_mock()
    except Exception as e:
        print(f"  [FAIL] GraphFactory test: {e}")
        import traceback; traceback.print_exc()
        results["factory_v2"] = False

    # Test 3: Runtime type branching
    try:
        results["runtime_v2"] = test_runtime_type_branching()
    except Exception as e:
        print(f"  [FAIL] Runtime test: {e}")
        import traceback; traceback.print_exc()
        results["runtime_v2"] = False

    # Test 3b: Tacit extracted-attributes postprocess
    try:
        results["tacit_postprocess_v2"] = test_tacit_postprocess_extracted_attributes()
    except Exception as e:
        print(f"  [FAIL] Tacit postprocess test: {e}")
        import traceback; traceback.print_exc()
        results["tacit_postprocess_v2"] = False

    # Test 4: Full generation (optional)
    try:
        results["full_generation_v2"] = test_full_dsag_generation(skip_if_no_key=True)
    except Exception as e:
        print(f"  [FAIL] Full generation test: {e}")
        import traceback; traceback.print_exc()
        results["full_generation_v2"] = False

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, passed_flag in results.items():
        status = "PASS" if passed_flag else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
