import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class GraphEmbeddingIndex:
    def __init__(self, graph_path: str):
        with open(graph_path) as f:
            self.graph = json.load(f)
        
        # Preload embeddings into a matrix
        self.node_ids = []
        self.embedding_matrix = []
        
        for node in self.graph['nodes']:
            if 'embedding' in node:
                self.node_ids.append(node['id'])
                self.embedding_matrix.append(node['embedding'])
        
        self.embedding_matrix = np.array(self.embedding_matrix)
    
    def find_similar_nodes(self, query: str, top_k: int = 5) -> List[dict]:
        """Find most relevant nodes for a query"""
        query_embed = st_model.encode(query).reshape(1, -1)
        
        # Compute similarities in bulk (faster than looping)
        similarities = cosine_similarity(
            query_embed,
            self.embedding_matrix
        )[0]
        
        # Get top matches with metadata
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [
            {
                'node_id': self.node_ids[i],
                'similarity': float(similarities[i]),
                **self._get_node_metadata(self.node_ids[i])
            }
            for i in top_indices
        ]
    
    def _get_node_metadata(self, node_id: str) -> dict:
        """Helper to fetch additional node info"""
        for node in self.graph['nodes']:
            if node['id'] == node_id:
                return {
                    'type': node.get('type'),
                    'repo': node.get('repo'),
                    'responsibility': node.get('responsibility', '')
                }
        return {}