def answer_question(question: str, graph: nx.DiGraph, llm_api, technical: bool = False) -> str:
    # Step 1: Find seed nodes via semantic similarity
    seed_nodes = find_semantic_matches(question, graph)
    
    # Step 2: Expand to related nodes in the graph
    relevant_nodes = expand_via_graph(seed_nodes, graph)
    
    # Step 3: Retrieve full context for these nodes
    context = build_context(relevant_nodes, graph)
    
    # Step 4: Generate answer with LLM
    return generate_answer(question, context, llm_api, technical)