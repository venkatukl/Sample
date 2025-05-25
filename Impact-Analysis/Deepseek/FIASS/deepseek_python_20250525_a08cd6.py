def build_context(nodes: Set[str], graph: nx.DiGraph) -> str:
    """Create a natural language context from graph nodes and edges"""
    context = []
    
    for node in nodes:
        # Add node description
        node_data = graph.nodes[node]
        context.append(f"Component: {node}\nType: {node_data['type']}\nRepo: {node_data['repo']}")
        
        # Add its relationships
        for src, tgt, data in graph.out_edges(node, data=True):
            context.append(f"  - {data['type']} → {tgt} ({data.get('label', '')})")
    
    return "\n\n".join(context)