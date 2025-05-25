import networkx as nx
from typing import Dict, List

class EnhancedDependencyGraph:
    def __init__(self, graph_path: str):
        with open(graph_path) as f:
            self.raw_graph = json.load(f)
        
        # Create NetworkX graph for traversal
        self.nx_graph = nx.DiGraph()
        for node in self.raw_graph["nodes"]:
            self.nx_graph.add_node(node["id"], **node)
        
        for edge in self.raw_graph["links"]:
            self.nx_graph.add_edge(edge["source"], edge["target"], **edge)
        
        # Build auxiliary indices
        self._build_indices()
    
    def _build_indices(self):
        """Create fast lookup structures"""
        self.class_to_repo = {}  # "com.example.UserService" -> "repo1"
        self.repo_services = defaultdict(list)  # "repo1" -> [list of services]
        
        for node in self.nx_graph.nodes():
            if ":" in node:
                repo, class_fqn = node.split(":", 1)
                self.class_to_repo[class_fqn] = repo
                self.repo_services[repo].append(class_fqn)
    
    def get_upstream_dependencies(self, class_fqn: str, depth=2) -> List[Dict]:
        """Find all classes that depend on this one"""
        results = []
        for source, _ in nx.bfs_edges(self.nx_graph, class_fqn, depth_limit=depth, reverse=True):
            results.append({
                "class": source,
                "relationship": self.nx_graph.edges[source, class_fqn]["type"]
            })
        return results
    
    def get_downstream_impact(self, class_fqn: str) -> List[Dict]:
        """Find all classes this one depends on"""
        return [
            {"class": target, "relationship": data["type"]}
            for _, target, data in self.nx_graph.out_edges(class_fqn, data=True)
        ]