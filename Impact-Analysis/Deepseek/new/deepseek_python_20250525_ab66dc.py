class GraphAwareRetriever:
    def __init__(self, graph: EnhancedDependencyGraph, st_model: SentenceTransformer):
        self.graph = graph
        self.st_model = st_model
        self.embedding_cache = {}  # Cache node embeddings
    
    def get_node_embedding(self, node_id: str) -> np.ndarray:
        """Embed class names/responsibilities"""
        if node_id not in self.embedding_cache:
            node_data = self.graph.nx_graph.nodes[node_id]
            text = f"{node_id}: {node_data.get('responsibility', '')}"
            self.embedding_cache[node_id] = self.st_model.encode(text)
        return self.embedding_cache[node_id]
    
    def semantic_search(self, query: str, top_k=5) -> List[Dict]:
        """Find relevant nodes via semantic similarity"""
        query_embedding = self.st_model.encode(query)
        scores = []
        
        for node_id in self.graph.nx_graph.nodes():
            node_embedding = self.get_node_embedding(node_id)
            similarity = np.dot(query_embedding, node_embedding)
            scores.append((node_id, similarity))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    
    def expand_with_relationships(self, node_ids: List[str]) -> Dict:
        """Augment results with graph context"""
        result = {"nodes": [], "edges": []}
        
        for node_id in node_ids:
            # Add the node itself
            result["nodes"].append(self.graph.nx_graph.nodes[node_id])
            
            # Add immediate relationships
            for _, neighbor, data in self.graph.nx_graph.edges(node_id, data=True):
                result["edges"].append({
                    "source": node_id,
                    "target": neighbor,
                    "type": data["type"]
                })
        
        return result