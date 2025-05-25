import javalang
import os
import yaml
import requests
import json
from collections import defaultdict

def parse_config_file(file_path):
    """
    Parse configuration files (.yml or .properties) to extract dependencies.
    
    Args:
        file_path (str): Path to the configuration file.
    Returns:
        dict: Parsed configuration data.
    """
    config = {}
    try:
        if file_path.endswith('.yml') or file_path.endswith('.yaml'):
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        elif file_path.endswith('.properties'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        config[key.strip()] = value.strip()
        return config
    except Exception as e:
        print(f"Error parsing config file {file_path}: {e}")
        return {}

def extract_comments(file_path):
    """
    Extract comments from a Java file.
    
    Args:
        file_path (str): Path to the Java file.
    Returns:
        list: List of comment strings.
    """
    comments = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
            tree = javalang.parse.parse(code)
            for path, node in tree:
                if isinstance(node, javalang.tree.Documentation) or isinstance(node, javalang.tree.Comment):
                    comments.append(node.text.strip())
        return comments
    except Exception as e:
        print(f"Error extracting comments from {file_path}: {e}")
        return []

def query_llm_for_dependencies(code_snippet, config_data, comments, api_endpoint="https://your-llm-api-endpoint"):
    """
    Use LLM to infer dynamic dependencies from code, config, and comments.
    
    Args:
        code_snippet (str): Code containing Feign clients or other dynamic constructs.
        config_data (dict): Parsed configuration data.
        comments (list): List of code comments.
        api_endpoint (str): LLM API endpoint.
    Returns:
        dict: Inferred dependencies (nodes and edges).
    """
    prompt = f"""
    Analyze the following to identify dynamic dependencies (e.g., Feign client target services, database connections, message queues):
    
    Code:
    {code_snippet}
    
    Configuration:
    {json.dumps(config_data, indent=2)}
    
    Comments:
    {json.dumps(comments, indent=2)}
    
    Return a JSON object with:
    - nodes: List of new nodes (e.g., services, databases, queues).
    - edges: List of new edges (e.g., API calls, DB access).
    
    Example output:
    {
      "nodes": [{"id": "payment-service", "type": "service", "repo": "inferred"}],
      "edges": [{"source": "repo_1:UserService", "target": "payment-service", "type": "api_call"}]
    }
    """
    
    payload = {'prompt': prompt, 'max_tokens': 1000}
    try:
        response = requests.post(api_endpoint, json=payload)
        response.raise_for_status()
        return json.loads(response.json().get('text', '{}'))
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return {'nodes': [], 'edges': []}

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
                                'endpoint': ann.element.get('value', '') if ann.element else '',
                                'method': method.name
                            })
                        # Detect MQ dependencies
                        if ann.name == 'JmsListener':
                            result['dependencies']['mq'].append({
                                'destination': ann.element.get('destination', '') if ann.element else ''
                            })
                
                # Detect database dependencies
                if 'Entity' in class_info['annotations'] or 'Repository' in class_info['annotations']:
                    result['dependencies']['db'].append({'class': node.name})
                
                # Detect Feign clients
                if 'FeignClient' in class_info['annotations']:
                    for ann in node.annotations:
                        if ann.name == 'FeignClient':
                            service_name = ann.element.get('name', '') if ann.element else ''
                            result['dependencies']['api_calls'].append({
                                'feign_client': service_name,
                                'class': node.name
                            })
        
        return result
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def generate_dependency_graph(repo_paths, output_file="dependency_graph.json", api_endpoint="https://your-llm-api-endpoint"):
    """
    Generate a dependency graph from all Java and config files, using LLM for dynamic dependencies.
    
    Args:
        repo_paths (list): List of paths to cloned repositories.
        output_file (str): Path to save the JSON dependency graph.
        api_endpoint (str): LLM API endpoint.
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
        
        # Parse Java files
        for root, _, files in os.walk(repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.java'):
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
                                if 'endpoint' in dep:
                                    graph['edges'].append({
                                        'source': f"{repo_name}:{cls['name']}",
                                        'target': f"api:{dep['endpoint']}",
                                        'type': 'api_call'
                                    })
                                elif 'feign_client' in dep:
                                    # Use LLM to infer Feign client target
                                    comments = extract_comments(file_path)
                                    llm_deps = query_llm_for_dependencies(
                                        code_snippet=open(file_path, 'r', encoding='utf-8').read(),
                                        config_data={},
                                        comments=comments,
                                        api_endpoint=api_endpoint
                                    )
                                    graph['nodes'].extend(llm_deps.get('nodes', []))
                                    graph['edges'].extend([
                                        {**edge, 'source': f"{repo_name}:{cls['name']}" if edge['source'] == dep['feign_client'] else edge['source']}
                                        for edge in llm_deps.get('edges', [])
                                    ])
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
                
                # Parse configuration files
                elif file.endswith(('.yml', '.yaml', '.properties')):
                    config_data = parse_config_file(file_path)
                    comments = []  # No comments in config files, but could extend to parse nearby READMEs
                    llm_deps = query_llm_for_dependencies(
                        code_snippet='',
                        config_data=config_data,
                        comments=comments,
                        api_endpoint=api_endpoint
                    )
                    graph['nodes'].extend(llm_deps.get('nodes', []))
                    graph['edges'].extend(llm_deps.get('edges', []))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(graph, f, indent=2)
    
    return graph

# Example usage
# repo_paths = ["repos/repo_0_repo1", "repos/repo_1_repo2", "repos/repo_2_repo3"]
# graph = generate_dependency_graph(repo_paths, api_endpoint="https://your-llm-api-endpoint")