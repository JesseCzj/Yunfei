#!/usr/bin/env python3
"""
DSAG Visualizer and Testing Script

Generates a DSAG graph and outputs:
1. dsag_output.json - Full graph JSON for inspection
2. dsag_alignments.json - Alignment judgments (Agent C output)
3. dsag_visualization.html - Mermaid.js diagrams (open in browser)
4. Console output - Runtime analysis results

Usage:
    python visualize_dsag.py

Requirements:
    - Valid API keys in .env (OPENAI_API_KEY for embeddings)
    - Dependencies: pip install -r requirements.txt
"""

import json
import os
import sys
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed, using existing environment variables")


def sanitize_mermaid_id(text: str) -> str:
    """Convert text to a valid Mermaid node ID."""
    # Replace spaces and special chars with underscores
    result = ""
    for c in text:
        if c.isalnum():
            result += c
        else:
            result += "_"
    return result[:30]  # Limit length


def sanitize_mermaid_label(text: str) -> str:
    """Escape text for Mermaid labels."""
    # Escape quotes and special characters
    text = text.replace('"', "'")
    text = text.replace('<', "")
    text = text.replace('>', "")
    text = text.replace('&', "and")
    # Truncate long labels
    if len(text) > 40:
        text = text[:37] + "..."
    return text


def tree_to_mermaid(tree, title: str) -> str:
    """Convert a TaxonomyTree to Mermaid flowchart syntax."""
    lines = [f"flowchart TD"]
    lines.append(f"    subgraph {sanitize_mermaid_id(title)} [{title}]")
    
    # Add nodes and edges
    for node_id, node in tree.nodes.items():
        safe_id = sanitize_mermaid_id(node_id)
        safe_label = sanitize_mermaid_label(node.label)
        
        # Style based on layer
        if node.layer == "root":
            lines.append(f'    {safe_id}["{safe_label}"]')
        elif node.layer == "leaf":
            lines.append(f'    {safe_id}(["{safe_label}"])')
        else:
            lines.append(f'    {safe_id}["{safe_label}"]')
        
        # Add edge to parent
        if node.parent_id and node.parent_id in tree.nodes:
            parent_safe_id = sanitize_mermaid_id(node.parent_id)
            lines.append(f"    {parent_safe_id} --> {safe_id}")
    
    lines.append("    end")
    return "\n".join(lines)


def links_to_mermaid(graph) -> str:
    """Convert gap links to a Mermaid bipartite graph."""
    lines = ["flowchart LR"]
    
    # Expert leaves on left
    lines.append("    subgraph expert [Expert Pain Points]")
    expert_leaves = graph.expert_tree.get_leaves()
    for leaf in expert_leaves:
        safe_id = "E_" + sanitize_mermaid_id(leaf.id)
        safe_label = sanitize_mermaid_label(leaf.label)
        lines.append(f'    {safe_id}(["{safe_label}"])')
    lines.append("    end")
    
    # Researcher leaves on right
    lines.append("    subgraph researcher [Researcher Goals]")
    researcher_leaves = graph.researcher_tree.get_leaves()
    for leaf in researcher_leaves:
        safe_id = "R_" + sanitize_mermaid_id(leaf.id)
        safe_label = sanitize_mermaid_label(leaf.label)
        lines.append(f'    {safe_id}(["{safe_label}"])')
    lines.append("    end")
    
    # Add links
    for link in graph.links:
        exp_id = "E_" + sanitize_mermaid_id(link.expert_leaf_id)
        res_id = "R_" + sanitize_mermaid_id(link.researcher_leaf_id)
        relation = sanitize_mermaid_label(link.relation_type)
        lines.append(f'    {exp_id} -.->|"{relation}"| {res_id}')
    
    return "\n".join(lines)


def generate_html(expert_tree_mermaid: str, researcher_tree_mermaid: str, links_mermaid: str, metadata: dict) -> str:
    """Generate self-contained HTML with Mermaid diagrams."""
    alignment_stats = metadata.get("alignment_stats") or {}
    alignment_lines = ""
    if alignment_stats:
        alignment_lines = (
            f'<p><strong>Leaf Alignments:</strong> {alignment_stats.get("leaf_alignments", "N/A")}</p>'
            f'<p><strong>Concept Alignments:</strong> {alignment_stats.get("concept_alignments", "N/A")}</p>'
            f'<p><strong>Misaligned Leaves:</strong> {alignment_stats.get("misaligned_leaves", "N/A")}</p>'
        )
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DSAG Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4a90d9;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .metadata {{
            background: #fff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metadata p {{
            margin: 5px 0;
            color: #666;
        }}
        .diagram-container {{
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
        .mermaid {{
            display: flex;
            justify-content: center;
        }}
    </style>
</head>
<body>
    <h1>DSAG Visualization</h1>
    
    <div class="metadata">
        <p><strong>Topic:</strong> {metadata.get('topic', 'N/A')}</p>
        <p><strong>Generated:</strong> {metadata.get('created_at', 'N/A')}</p>
        <p><strong>Expert Leaves:</strong> {metadata.get('expert_leaves', 'N/A')}</p>
        <p><strong>Researcher Leaves:</strong> {metadata.get('researcher_leaves', 'N/A')}</p>
        <p><strong>Gap Links:</strong> {metadata.get('link_count', 'N/A')}</p>
        <p><strong>Construction Method:</strong> {metadata.get('construction_method', 'N/A')}</p>
        <p><strong>Templates Generated:</strong> {metadata.get('templates_generated', 'N/A')}</p>
        {alignment_lines}
    </div>
    
    <h2>Expert Taxonomy Tree (TExp)</h2>
    <div class="diagram-container">
        <pre class="mermaid">
{expert_tree_mermaid}
        </pre>
    </div>
    
    <h2>Researcher Taxonomy Tree (TRes)</h2>
    <div class="diagram-container">
        <pre class="mermaid">
{researcher_tree_mermaid}
        </pre>
    </div>
    
    <h2>Gap Links (Expert Pain Points to Researcher Goals)</h2>
    <div class="diagram-container">
        <pre class="mermaid">
{links_mermaid}
        </pre>
    </div>
    
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
</body>
</html>
"""
    return html


def _load_questionnaire_text() -> str:
    """Load questionnaire.docx from workspace root if present."""
    questionnaire_path = os.path.join(os.getcwd(), "questionnaire.docx")
    if not os.path.exists(questionnaire_path):
        return ""
    try:
        import docx
    except Exception:
        return ""
    try:
        document = docx.Document(questionnaire_path)
        return "\n".join([p.text for p in document.paragraphs if p.text])
    except Exception:
        return ""


def main():
    print("=" * 60)
    print("DSAG Visualizer and Testing Script")
    print("=" * 60)
    
    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key == "your_openai_api_key_here":
        print("\n[ERROR] OPENAI_API_KEY is required for embeddings.")
        print("Please set it in .env file.")
        sys.exit(1)
    
    # Sample inputs
    topic = "Virtual cell AI models for gene perturbation experiments (CellForge)"
    researcher_bg = "HCI/visualization researcher studying explainable AI for virtual cell models and gene prediction workflows"
    expert_bg = "Molecular biologist conducting gene perturbation experiments and evaluating virtual cell model reliability"
    generate_templates = os.getenv("DSAG_GENERATE_TEMPLATES", "true").lower() in {"1", "true", "yes"}
    questionnaire = _load_questionnaire_text()
    
    print(f"\n[Config]")
    print(f"  Topic: {topic}")
    print(f"  Researcher BG: {researcher_bg}")
    print(f"  Expert BG: {expert_bg}")
    print(f"  Generate Templates: {generate_templates}")
    if questionnaire:
        print(f"  Questionnaire: loaded ({len(questionnaire)} chars)")
    else:
        print(f"  Questionnaire: not found or empty")
    
    # Import DSAG modules
    print("\n[Step 1] Importing DSAG modules...")
    try:
        from dsag.factory import GraphFactory
        from dsag.schema import DSAGGraph
        from dsag.embedding_index import build_embedding_index
        from dsag.runtime import RuntimeEngine
        print("  Imports successful!")
    except ImportError as e:
        print(f"  [ERROR] Failed to import DSAG modules: {e}")
        print("  Make sure you're in the project directory and dependencies are installed.")
        sys.exit(1)
    
    # Generate DSAG graph
    print("\n[Step 2] Generating DSAG graph (this may take 30-60 seconds)...")
    start_time = datetime.now()
    try:
        factory = GraphFactory()
        print("  Agent A: Generating expert tree...")
        expert_tree = factory.generate_expert_tree(topic, expert_bg)
        print(f"  Expert tree: {len(expert_tree.nodes)} nodes, {len(expert_tree.get_leaves())} leaves")
        
        print("  Agent B: Generating researcher tree...")
        researcher_tree = factory.generate_researcher_tree(topic, researcher_bg, questionnaire)
        print(f"  Researcher tree: {len(researcher_tree.nodes)} nodes, {len(researcher_tree.get_leaves())} leaves")
        
        print("  Agent C + Math Algorithm: Generating gap links...")
        links, alignments = factory.generate_links(
            topic,
            expert_tree,
            researcher_tree,
            generate_templates=generate_templates,
        )
        
        graph = DSAGGraph(
            topic=topic,
            researcher_bg=researcher_bg,
            expert_bg=expert_bg,
            expert_tree=expert_tree,
            researcher_tree=researcher_tree,
            links=links,
            metadata={
                "created_at": datetime.utcnow().isoformat(),
                "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
                "expert_leaves": len(expert_tree.get_leaves()),
                "researcher_leaves": len(researcher_tree.get_leaves()),
                "link_count": len(links),
                "alignment_stats": {
                    "leaf_alignments": len(alignments.leaf_alignments),
                    "concept_alignments": len(alignments.concept_alignments),
                    "misaligned_leaves": len(alignments.get_misaligned_leaf_pairs()),
                },
                "construction_method": "math-based",
                "templates_generated": generate_templates,
            },
        )
        duration = (datetime.now() - start_time).total_seconds()
        print(f"  Graph generated in {duration:.1f}s")
        print(f"  Gap links: {len(graph.links)}")
        print(f"  Leaf alignments: {len(alignments.leaf_alignments)}")
        print(f"  Concept alignments: {len(alignments.concept_alignments)}")
    except Exception as e:
        print(f"  [ERROR] Failed to generate graph: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save JSON output
    print("\n[Step 3] Saving graph to dsag_output.json...")
    try:
        with open("dsag_output.json", "w", encoding="utf-8") as f:
            f.write(graph.to_json())
        print("  Saved successfully!")
    except Exception as e:
        print(f"  [ERROR] Failed to save JSON: {e}")
    
    # Save alignments
    print("\n[Step 4] Saving alignments to dsag_alignments.json...")
    try:
        with open("dsag_alignments.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(alignments.to_dict(), indent=2, ensure_ascii=False))
        print("  Saved successfully!")
    except Exception as e:
        print(f"  [ERROR] Failed to save alignments: {e}")
    
    # Generate Mermaid diagrams
    print("\n[Step 5] Generating Mermaid diagrams...")
    try:
        expert_mermaid = tree_to_mermaid(graph.expert_tree, "Expert_Tree")
        researcher_mermaid = tree_to_mermaid(graph.researcher_tree, "Researcher_Tree")
        links_mermaid = links_to_mermaid(graph)
        print("  Diagrams generated!")
    except Exception as e:
        print(f"  [ERROR] Failed to generate diagrams: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Generate HTML
    print("\n[Step 6] Generating HTML visualization...")
    try:
        metadata = {
            "topic": topic,
            **graph.metadata
        }
        html = generate_html(expert_mermaid, researcher_mermaid, links_mermaid, metadata)
        with open("dsag_visualization.html", "w", encoding="utf-8") as f:
            f.write(html)
        print("  Saved to dsag_visualization.html")
        print("  Open this file in a browser to view the diagrams!")
    except Exception as e:
        print(f"  [ERROR] Failed to generate HTML: {e}")
    
    # Build embeddings and test runtime
    print("\n[Step 7] Building embedding index...")
    try:
        embedding_index = build_embedding_index(graph)
        print(f"  Expert embeddings: {len(embedding_index.expert_leaf_embeddings)}")
        print(f"  Researcher embeddings: {len(embedding_index.researcher_leaf_embeddings)}")
    except Exception as e:
        print(f"  [ERROR] Failed to build embeddings: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test runtime analysis
    print("\n[Step 8] Testing runtime analysis...")
    try:
        engine = RuntimeEngine(graph, embedding_index)
        
        # Sample conversation turn
        test_question = "How do you evaluate whether a virtual cell model like CellForge is reliable? Which metrics influence your judgment?"
        test_answer = "I look at how well predictions match known perturbation effects and whether the model is consistent across replicates, but sometimes the confidence scores feel hard to trust without more context."
        
        print(f"\n  Test Question: {test_question}")
        print(f"  Test Answer: {test_answer}")
        
        context_summary = ""
        analysis = engine.analyze_turn(test_question, test_answer, context_summary=context_summary, interview_timeline=[])
        
        print("\n  === Analysis Results ===")
        
        # Located positions
        located = analysis.located
        if located.best_expert_leaf_id:
            expert_node = graph.expert_tree.get_node(located.best_expert_leaf_id)
            print(f"\n  Expert Position:")
            print(f"    Leaf: {expert_node.label if expert_node else 'N/A'}")
            print(f"    Confidence: {located.expert_confidence:.2f}")
        
        if located.best_researcher_leaf_id:
            res_node = graph.researcher_tree.get_node(located.best_researcher_leaf_id)
            print(f"\n  Researcher Position:")
            print(f"    Leaf: {res_node.label if res_node else 'N/A'}")
            print(f"    Confidence: {located.researcher_confidence:.2f}")
        
        # Divergence
        if analysis.divergence:
            print(f"\n  Divergence:")
            print(f"    LCA Layer: {analysis.divergence.lca_layer}")
            print(f"    Expert Branch: {analysis.divergence.expert_branch}")
            print(f"    Researcher Branch: {analysis.divergence.researcher_branch}")
        
        # Assistance (type-specific payload + follow-ups)
        if analysis.assistance:
            assistance = analysis.assistance
            print(f"\n  Assistance:")
            print(f"    Relation Type: {assistance.relation_type}")
            print(f"    Payload keys: {list(assistance.payload.keys())}")
            for key, val in assistance.payload.items():
                val_str = str(val)[:120]
                print(f"      {key}: {val_str}...")
        
        print("\n  Runtime analysis completed!")
        
    except Exception as e:
        print(f"  [ERROR] Failed to run analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("DSAG Testing Complete!")
    print("=" * 60)
    print("\nOutput files:")
    print("  1. dsag_output.json - Full graph JSON")
    print("  2. dsag_alignments.json - Alignment judgments")
    print("  3. dsag_visualization.html - Open in browser to view diagrams")
    print("\nNext steps:")
    print("  - Inspect dsag_output.json to see the full graph structure")
    print("  - Open dsag_visualization.html in a browser")
    print("  - Modify the test_question/test_answer to try different scenarios")


if __name__ == "__main__":
    main()
