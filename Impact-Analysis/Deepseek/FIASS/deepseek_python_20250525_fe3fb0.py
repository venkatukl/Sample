def answer_question(question: str):
    # Step 1: Semantic search with FAISS/SentenceTransformer
    query_embedding = st_model.encode(question)
    similar_nodes = faiss_index.search(query_embedding, k=3)
    
    # Step 2: Graph analysis with NetworkX
    impacted_components = []
    for node in similar_nodes:
        impacted_components.extend(
            nx.bfs_tree(graph, node, depth_limit=2).nodes()
        )
    
    # Step 3: Synthesize answer using LLM
    context = get_code_context(impacted_components)
    return llm.generate_answer(question, context)