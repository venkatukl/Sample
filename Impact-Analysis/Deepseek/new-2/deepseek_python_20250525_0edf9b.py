# Add this to your existing graph builder
graph = DependencyGraphBuilder()
graph.add_analysis_result(analysis)  # Your existing code
enhanced_graph = add_embeddings_to_graph(graph.to_json())