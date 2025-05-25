import javalang
import os
import json
from collections import defaultdict

def parse_java_file(file_path):
    """
    Parse a Java file to extract classes, methods, and dependencies.
    
    Args:
        file_path (str): Path to the Java file.
    Returns:
        dict: Parsed information (classes, methods, annotations, dependencies).
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    try:
        tree = javalang.parse.parse(code)
        result = {
            'file': file_path,
            'classes': [],
            'dependencies': {'api_calls': [], 'db': [], 'mq': []}
        }
        
        for path, node in tree:
            if isinstance(node, javalang.tree.ClassDeclaration):
                class_info = {'name': node.name, 'methods': [], 'annotations': [ann.name for ann in node.annotations]}
                result['classes'].append(class_info)
                
                for method in node.methods:
                    method_info = {
                        'name': method.name,
                        'annotations': [ann.name for ann in method.annotations],
                        'parameters': [param.type.name for param in method.parameters]
                    }
                    class_info['methods'].append(method_info)
                    
                    # Detect API endpoints
                    for ann in method.annotations:
                        if ann.name in ['GetMapping', 'PostMapping', 'RequestMapping']:
                            result['dependencies']['api_calls'].append({
                                'endpoint': ann.element.get('value', ''),
                                'method': method.name
                            })
                        # Detect MQ dependencies
                        if ann.name == 'JmsListener':
                            result['dependencies']['mq'].append({
                                'destination': ann.element.get('destination', '')
                            })
                
                # Detect database dependencies
                if 'Entity' in class_info['annotations'] or 'Repository' in class_info['annotations']:
                    result['dependencies']['db'].append({'class': node.name})
                    
        return result
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def generate_dependency_graph(repo_paths, output_file="dependency_graph.json"):
    """
    Generate a dependency graph from all Java files in the repositories.
    
    Args:
        repo_paths (list): List of paths to cloned repositories.
        output_file (str): Path to save the JSON dependency graph.
    Returns:
        dict: Dependency graph.
    """
    graph = {
        'nodes': [],
        'edges': [],
        'repos': {}
    }
    
    for repo_path in repo_paths:
        repo_name = os.path.basename(repo_path)
        graph['repos'][repo_name] = []
        
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.java'):
                    file_path = os.path.join(root, file)
                    parsed = parse_java_file(file_path)
                    if parsed:
                        graph['repos'][repo_name].append(parsed)
                        for cls in parsed['classes']:
                            graph['nodes'].append({
                                'id': f"{repo_name}:{cls['name']}",
                                'type': 'class',
                                'repo': repo_name
                            })
                            for dep in parsed['dependencies']['api_calls']:
                                # Assume API calls to other repos based on URL patterns
                                graph['edges'].append({
                                    'source': f"{repo_name}:{cls['name']}",
                                    'target': f"api:{dep['endpoint']}",
                                    'type': 'api_call'
                                })
                            for dep in parsed['dependencies']['db']:
                                graph['edges'].append({
                                    'source': f"{repo_name}:{cls['name']}",
                                    'target': f"db:{dep['class']}",
                                    'type': 'db_access'
                                })
                            for dep in parsed['dependencies']['mq']:
                                graph['edges'].append({
                                    'source': f"{repo_name}:{cls['name']}",
                                    'target': f"mq:{dep['destination']}",
                                    'type': 'mq_access'
                                })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(graph, f, indent=2)
    
    return graph

# Example usage
# repo_paths = ["repos/repo_0_repo1", "repos/repo_1_repo2", "repos/repo_2_repo3"]
# graph = generate_dependency_graph(repo_paths)