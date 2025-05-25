def find_semantic_matches(query: str, graph: nx.DiGraph, top_k: int = 3) -> List[str]:
    """Find graph nodes most relevant to the query using embeddings"""
    query_embed = st_model.encode(query)
    
    scores = []
    for node in graph.nodes():
        # Use node's name + type as embedding text (precomputed during graph build)
        node_embed = graph.nodes[node].get('embedding')  
        similarity = cosine_similarity(query_embed, node_embed)
        scores.append((node, similarity))
    
    return [node for node, _ in sorted(scores, key=lambda x: -x[1])[:top_k]]