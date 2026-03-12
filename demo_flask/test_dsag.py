#!/usr/bin/env python3
"""
DSAG Smoke Test Script

Tests the DSAG (Dual-Tower Semantic Alignment Graph) backend components.
Run with: python test_dsag.py

Requires:
- Valid OPENAI_API_KEY for embeddings
- Valid LLM_PROVIDER API key (OpenAI or DeepSeek)
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
    """Test DSAG schema classes."""
    print("\n=== Testing Schema ===")
    
    from dsag.schema import (
        DSAGNode, TaxonomyTree, GapLink, DSAGGraph,
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
    )
    assert node.id == "test_node"
    assert node.is_leaf()
    print("  [OK] Node creation")
    
    # Test LCA computation
    path1 = ["Root", "Perspective A", "Category 1", "Leaf 1"]
    path2 = ["Root", "Perspective A", "Category 2", "Leaf 2"]
    lca = compute_lca_layer(path1, path2)
    assert lca == Layer.L1.value, f"Expected L1, got {lca}"
    print("  [OK] LCA computation")
    
    # Test serialization
    node_dict = node.to_dict()
    node_restored = DSAGNode.from_dict(node_dict)
    assert node_restored.label == node.label
    print("  [OK] Node serialization")
    
    print("  Schema tests passed!")
    return True


def test_graph_factory_mock():
    """Test GraphFactory with minimal mocking (no API calls)."""
    print("\n=== Testing GraphFactory (structure only) ===")
    
    from dsag.factory import _build_expert_tree, _build_researcher_tree
    
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
                    }
                ]
            }
        ]
    }
    
    # Mock LLM response for researcher
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
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    # Build trees
    topic = "AI-assisted Medical Diagnosis"
    expert_tree = _build_expert_tree(mock_expert_response, topic)
    researcher_tree = _build_researcher_tree(mock_researcher_response, topic)
    
    # Verify structure
    assert len(expert_tree.get_leaves()) == 2, f"Expected 2 expert leaves, got {len(expert_tree.get_leaves())}"
    assert len(researcher_tree.get_leaves()) == 1, f"Expected 1 researcher leaf, got {len(researcher_tree.get_leaves())}"
    print(f"  [OK] Expert tree: {len(expert_tree.nodes)} nodes, {len(expert_tree.get_leaves())} leaves")
    print(f"  [OK] Researcher tree: {len(researcher_tree.nodes)} nodes, {len(researcher_tree.get_leaves())} leaves")
    
    # Check attributes
    intuition_nodes = [n for n in expert_tree.get_leaves() if n.attributes]
    assert len(intuition_nodes) == 1, "Should have 1 node with attributes"
    print(f"  [OK] Intuition node has attributes: {intuition_nodes[0].attributes}")
    
    print("  GraphFactory structure tests passed!")
    return True


def test_full_dsag_generation(skip_if_no_key=True):
    """Test full DSAG generation with real API calls."""
    print("\n=== Testing Full DSAG Generation ===")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        if skip_if_no_key:
            print("  [SKIP] No valid OPENAI_API_KEY, skipping full generation test")
            return True
        else:
            raise ValueError("OPENAI_API_KEY required for full test")
    
    from dsag import create_dsag_graph, build_embedding_index, RuntimeEngine
    
    # Simple test case
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
    
    # Test embeddings
    print("  Building embeddings index...")
    index = build_embedding_index(graph)
    print(f"  [OK] Embeddings built: {len(index.expert_leaf_embeddings)} expert, {len(index.researcher_leaf_embeddings)} researcher")
    
    # Test runtime
    print("  Testing runtime analysis...")
    engine = RuntimeEngine(graph, index)
    
    test_question = "How do you interpret the AI's confidence score?"
    test_answer = "I look at the highlighted areas, but honestly sometimes I just ignore it when it doesn't match my intuition."
    
    analysis = engine.analyze_turn(test_question, test_answer, context_summary="")
    
    print(f"  [OK] Analysis completed")
    print(f"       Expert position: {analysis.located.best_expert_leaf_id} (score={analysis.located.expert_confidence:.2f})")
    print(f"       Researcher position: {analysis.located.best_researcher_leaf_id} (score={analysis.located.researcher_confidence:.2f})")
    
    if analysis.divergence:
        print(f"       Divergence: LCA={analysis.divergence.lca_layer}")
    
    if analysis.assistance:
        print(f"       Payload keys: {list(analysis.assistance.payload.keys())}")
    
    print("  Full DSAG generation test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("DSAG Smoke Test")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Schema
    try:
        results["schema"] = test_schema()
    except Exception as e:
        print(f"  [FAIL] Schema test: {e}")
        results["schema"] = False
    
    # Test 2: GraphFactory structure
    try:
        results["factory_mock"] = test_graph_factory_mock()
    except Exception as e:
        print(f"  [FAIL] GraphFactory mock test: {e}")
        results["factory_mock"] = False
    
    # Test 3: Full generation (optional, requires API key)
    try:
        results["full_generation"] = test_full_dsag_generation(skip_if_no_key=True)
    except Exception as e:
        print(f"  [FAIL] Full generation test: {e}")
        results["full_generation"] = False
    
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
