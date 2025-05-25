class ConsolidatedGraph:
    def __init__(self):
        self.nodes = set()  # Format: "repo:package.Class"
        self.edges = []     # Format: {"source": "x", "target": "y", "type": "..."}
    
    def add_dependencies(self, analysis: dict):
        source_id = f"{analysis['repository']}:{analysis['package']}.{analysis['class_name']}"
        self.nodes.add(source_id)
        
        # Intra-repo dependencies
        for dep in analysis['dependencies'].get('intra_repo', []):
            target_id = f"{analysis['repository']}:{analysis['package']}.{dep['target_class']}"
            self._add_edge(source_id, target_id, dep['type'])
        
        # Inter-repo dependencies
        for dep in analysis['dependencies'].get('inter_repo', []):
            target_id = f"{dep['target_repo']}:{dep['target_service']}"  # Assumes service=class
            self._add_edge(source_id, target_id, dep['protocol'])
    
    def _add_edge(self, source: str, target: str, relationship: str):
        self.edges.append({
            "source": source,
            "target": target,
            "type": relationship
        })
        self.nodes.add(target)  # Ensure target exists in graph
    
    def to_json(self):
        return {
            "nodes": [{"id": n} for n in self.nodes],
            "links": self.edges
        }