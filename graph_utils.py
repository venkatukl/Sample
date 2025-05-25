import json
from collections import defaultdict

def load_dependency_graph(graph_file="dependency_graph.json"):
    """
    Load the dependency graph from a JSON file.
    
    Args:
        graph_file (str): Path to the dependency graph JSON.
    Returns:
        dict: Loaded dependency graph.
    """
    try:
        with open(graph_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading dependency graph: {e}")
        return {'nodes': [], 'edges': [], 'repos': {}}

def get_relevant_graph_data(graph, query):
    """
    Extract relevant nodes and edges from the dependency graph based on the query.
    
    Args:
        graph (dict): Dependency graph.
        query (str): User query.
    Returns:
        dict: Relevant nodes and edges.
    """
    relevant_data = {'nodes': [], 'edges': []}
    query_lower = query.lower()
    
    # Identify relevant nodes (e.g., classes, APIs, databases)
    for node in graph['nodes']:
        if query_lower in node['id'].lower() or query_lower in node.get('type', '').lower():
            relevant_data['nodes'].append(node)
    
    # Identify relevant edges (e.g., API calls, DB access)
    for edge in graph['edges']:
        if any(query_lower in edge[field].lower() for field in ['source', 'target', 'type']):
            relevant_data['edges'].append(edge)
            # Include source and target nodes
            for node in graph['nodes']:
                if node['id'] in [edge['source'], edge['target']] and node not in relevant_data['nodes']:
                    relevant_data['nodes'].append(node)
    
    return relevant_data

# Example usage
# graph = load_dependency_graph()
# query = "UserService API"
# relevant_data = get_relevant_graph_data(graph, query)