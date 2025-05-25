# After constructing the raw graph
enhanced_graph = add_embeddings_to_graph(raw_graph)

# Now save to JSON
with open('dependency_graph.json', 'w') as f:
    json.dump(enhanced_graph, f)