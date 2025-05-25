# In ConsolidatedGraph
def detect_cycles(self):
    import networkx as nx
    G = nx.DiGraph()
    G.add_edges_from((e["source"], e["target"]) for e in self.edges)
    return list(nx.simple_cycles(G))