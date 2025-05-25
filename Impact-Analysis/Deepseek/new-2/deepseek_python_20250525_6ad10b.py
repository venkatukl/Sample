from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize embedding model (same as your retriever)
st_model = SentenceTransformer('all-MiniLM-L6-v2')

def add_embeddings_to_graph(graph_data: dict) -> dict:
    """Add embeddings to each node in the graph"""
    for node in graph_data['nodes']:
        # Create descriptive text for embedding
        node_text = f"""
        Component: {node['id']}
        Type: {node.get('type', 'class')}
        Repository: {node.get('repo', 'unknown')}
        Responsibility: {node.get('responsibility', '')}
        """
        
        # Generate and store embedding (as list for JSON serialization)
        node['embedding'] = st_model.encode(node_text).tolist()
    
    return graph_data