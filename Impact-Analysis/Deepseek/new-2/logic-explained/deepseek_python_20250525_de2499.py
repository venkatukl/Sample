# Step 1: Analyze file with LLM
analysis = llm_analyze_file(java_file)

# Step 2: Add to graph with enriched metadata
graph.add_node(
    id=generate_node_id(analysis),
    type=infer_node_type(java_file.content),  # From @Annotations
    repo=repo_name,
    responsibility=analysis["responsibility"],  # From LLM
    embedding=generate_embedding(analysis)     # Using responsibility + type
)

# Step 3: When querying, get high-quality matches
query = "Where are payments validated?"
results = graph.search(
    query_text=query,
    use_fields=["responsibility", "type"]  # Search these embeddings
)