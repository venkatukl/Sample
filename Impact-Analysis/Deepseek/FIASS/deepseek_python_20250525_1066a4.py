# Precompute during graph build
for node in graph.nodes():
    graph.nodes[node]['embedding'] = st_model.encode(
        f"{node} {graph.nodes[node]['type']}"
    )

# Cache frequent traversals
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_related_nodes(node: str) -> Set[str]:
    return set(nx.descendants(graph, node)) | set(nx.ancestors(graph, node))