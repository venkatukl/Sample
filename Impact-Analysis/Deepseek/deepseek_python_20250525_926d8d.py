from collections import defaultdict

class DependencyGraphBuilder:
    def __init__(self):
        self.graph = {
            'nodes': [],
            'edges': [],
            'repositories': set(),
            'inter_repo_edges': []
        }
        self.node_index = {}
    
    def add_analysis_result(self, analysis: Dict):
        """Add a single file's analysis to the graph"""
        if not analysis:
            return
        
        # Add repository
        repo_name = analysis.get('repository', 'unknown')
        self.graph['repositories'].add(repo_name)
        
        # Create node ID
        node_id = f"{analysis['package']}.{analysis['class_name']}" if analysis['package'] else analysis['class_name']
        node_id = f"{repo_name}:{node_id}"
        
        # Add node if not exists
        if node_id not in self.node_index:
            node = {
                'id': node_id,
                'label': analysis['class_name'],
                'package': analysis['package'],
                'repository': repo_name,
                'type': 'interface' if analysis.get('is_interface', False) else 'class',
                'responsibility': analysis.get('primary_responsibility', '')
            }
            self.graph['nodes'].append(node)
            self.node_index[node_id] = len(self.graph['nodes']) - 1
        
        # Add dependencies as edges
        for dep in analysis.get('external_dependencies', []):
            dep_id = self._get_dependency_id(dep, repo_name)
            edge = {
                'source': node_id,
                'target': dep_id,
                'type': dep.get('type', 'class'),
                'relationship': 'depends_on'
            }
            self.graph['edges'].append(edge)
        
        # Add inter-repo dependencies
        for inter_dep in analysis.get('inter_repo_dependencies', []):
            self.graph['inter_repo_edges'].append({
                'source': node_id,
                'target_repo': inter_dep['repo'],
                'target_service': inter_dep['service'],
                'type': 'inter_repository'
            })
    
    def _get_dependency_id(self, dep: Dict, source_repo: str) -> str:
        """Create or get ID for a dependency node"""
        dep_pkg = dep.get('package', '')
        dep_class = dep.get('class', dep.get('service', ''))
        
        if not dep_class:
            return 'unknown'
        
        dep_id = f"{dep_pkg}.{dep_class}" if dep_pkg else dep_class
        
        # Check if dependency is in another repo
        if 'repo' in dep:
            dep_id = f"{dep['repo']}:{dep_id}"
        else:
            dep_id = f"{source_repo}:{dep_id}"
        
        # Add dependency as node if not exists
        if dep_id not in self.node_index:
            node = {
                'id': dep_id,
                'label': dep_class,
                'package': dep_pkg,
                'repository': dep.get('repo', source_repo),
                'type': dep.get('type', 'class'),
                'responsibility': dep.get('responsibility', '')
            }
            self.graph['nodes'].append(node)
            self.node_index[dep_id] = len(self.graph['nodes']) - 1
        
        return dep_id
    
    def save_graph(self, output_path: Path):
        """Save the dependency graph to a JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.graph, f, indent=2)