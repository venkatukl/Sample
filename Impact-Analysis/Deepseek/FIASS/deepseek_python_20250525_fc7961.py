def expand_via_graph(seed_nodes: List[str], graph: nx.DiGraph, depth: int = 2) -> Set[str]:
    """Find connected nodes that provide meaningful context"""
    relevant_nodes = set()
    
    for node in seed_nodes:
        # Include the node itself
        relevant_nodes.add(node)
        
        # Add upstream/downstream dependencies
        relevant_nodes.update(nx.descendants(graph, node))  # Downstream
        relevant_nodes.update(nx.ancestors(graph, node))    # Upstream
        
        # Add siblings (nodes with common dependencies)
        for neighbor in graph.successors(node):
            relevant_nodes.update(graph.successors(neighbor))
    
    return relevant_nodes