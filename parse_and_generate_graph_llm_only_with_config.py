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

def chunk_file(file_path, max_tokens=1000):
    """
    Chunk a file into segments within the token limit.
    
    Args:
        file_path (str): Path to the file.
        max_tokens (int): Maximum tokens per chunk (approximate).
    Returns:
        list: List of (file_path, chunk_content) tuples.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Estimate tokens (1 token ~ 4 chars)
    tokens = len(content) // 4
    if tokens <= max_tokens:
        return [(file_path, content)]
    
    chunks = []
    lines = content.split('\n')
    current_chunk = []
    current_tokens = 0
    
    for line in lines:
        line_tokens = len(line) // 4
        if current_tokens + line_tokens > max_tokens:
            chunks.append((file_path, '\n'.join(current_chunk)))
            current_chunk = [line]
            current_tokens = line_tokens
        else:
            current_chunk.append(line)
            current_tokens += line_tokens
    
    if current_chunk:
        chunks.append((file_path, '\n'.join(current_chunk)))
    
    return chunks

def query_llm_for_dependencies(file_path, code_snippet, config_data, api_endpoint="https://your-llm-api-endpoint"):
    """
    Use LLM to infer dependencies from code and config, including @Configuration classes.
    
    Args:
        file_path (str): Path to the file (for context).
        code_snippet (str): Code or config content.
        config_data (dict): Parsed configuration data.
        api_endpoint (str): LLM API endpoint.
    Returns:
        dict: Inferred dependencies (nodes and edges).
    """
    prompt = f"""
    Analyze the following Spring Boot code or configuration file to identify dependencies, including:
    - Classes and services (e.g., annotated with @RestController, @Service, @Component, @Repository, @Configuration).
    - API endpoints (e.g., @GetMapping, @PostMapping).
    - Database connections (e.g., @Repository, spring.datasource.url).
    - Message queues (e.g., @JmsListener, spring.rabbitmq.host).
    - API calls via RestTemplate or WebClient (e.g., URLs from @Value or hardcoded).
    - Bean definitions in @Configuration classes (e.g., RestTemplate, TaskExecutor, CacheManager).
    
    File: {file_path}
    Code/Content:
    {code_snippet}
    
    Configuration:
    {json.dumps(config_data, indent=2)}
    
    Return a JSON object with:
    - nodes: List of nodes (e.g., classes, services, databases, queues). Use type: "configuration" for @Configuration classes.
    - edges: List of edges (e.g., API calls, DB access, MQ interactions).
    
    Example output:
    {
      "nodes": [
        {"id": "repo_1:AppConfig", "type": "configuration", "repo": "repo_1"},
        {"id": "payment-service", "type": "service", "repo": "inferred"},
        {"id": "db:User", "type": "database", "repo": "repo_1"}
      ],
      "edges": [
        {"source": "repo_1:AppConfig", "target": "payment-service", "type": "api_call"},
        {"source": "repo_1:UserService", "target": "db:User", "type": "db_access"}
      ]
    }
    """
    
    # Ensure prompt fits within ~4000-token limit (1 token ~ 4 chars)
    if len(prompt) // 4 > 3500:
        prompt = prompt[:14000] + "\n[Truncated due to length]"
    
    payload = {'prompt': prompt, 'max_tokens': 1000}
    try:
        response = requests.post(api_endpoint, json=payload)
        response.raise_for_status()
        return json.loads(response.json().get('text', '{}'))
    except Exception as e:
        print(f"Error querying LLM for {file_path}: {e}")
        return {'nodes': [], 'edges': []}

def generate_dependency_graph(repo_paths, output_file="dependency_graph.json", api_endpoint="https://your-llm-api-endpoint"):
    """
    Generate a dependency graph by sending all Java and config files to the LLM.
    
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
    
    # Collect all configuration data
    config_data = {}
    for repo_path in repo_paths:
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(('.yml', '.yaml', '.properties')):
                    file_path = os.path.join(root, file)
                    config_data.update(parse_config_file(file_path))
    
    # Process Java and config files
    for repo_path in repo_paths:
        repo_name = os.path.basename(repo_path)
        graph['repos'][repo_name] = []
        
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(('.java', '.yml', '.yaml', '.properties')):
                    file_path = os.path.join(root, file)
                    # Chunk large files
                    chunks = chunk_file(file_path, max_tokens=1000)
                    for chunk_path, chunk_content in chunks:
                        # Send to LLM
                        llm_deps = query_llm_for_dependencies(
                            file_path=chunk_path,
                            code_snippet=chunk_content,
                            config_data=config_data,
                            api_endpoint=api_endpoint
                        )
                        # Aggregate nodes and edges
                        for node in llm_deps.get('nodes', []):
                            if node not in graph['nodes']:
                                graph['nodes'].append(node)
                        for edge in llm_deps.get('edges', []):
                            if edge not in graph['edges']:
                                graph['edges'].append(edge)
                        graph['repos'][repo_name].append({
                            'file': chunk_path,
                            'dependencies': llm_deps
                        })
    
    # Save the graph
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(graph, f, indent=2)
    
    return graph

# Example usage
# repo_paths = ["repos/repo_0_repo1", "repos/repo_1_repo2", "repos/repo_2_repo3"]
# graph = generate_dependency_graph(repo_paths, api_endpoint="https://your-llm-api-endpoint")