# Store node embeddings in FAISS but keep relationships in NetworkX
faiss.add_vectors([graph.nodes[n]["embedding"] for n in nodes])