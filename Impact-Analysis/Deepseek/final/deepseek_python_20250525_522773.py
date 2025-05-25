import os
import json
import re
import numpy as np
from pathlib import Path
from git import Repo
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import networkx as nx

# Configuration
REPO_URLS = [
    "https://github.com/pauldragoslav/Spring-boot-Banking",
    "https://github.com/example/repo2",
    "https://github.com/example/repo3"
]
LLM_API_KEY = "your_api_key_here"  # Replace with your LLM API key

class DependencyAnalyzer:
    def __init__(self):
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.graph = nx.DiGraph()
        
    def clone_repos(self, repo_urls: List[str], clone_dir: str = "repos") -> List[Path]:
        """Clone repositories and return their paths"""
        os.makedirs(clone_dir, exist_ok=True)
        repo_paths = []
        for url in repo_urls:
            repo_name = url.split('/')[-1].replace('.git', '')
            repo_path = Path(clone_dir) / repo_name
            if not repo_path.exists():
                Repo.clone_from(url, repo_path)
            repo_paths.append(repo_path)
        return repo_paths

    def analyze_codebase(self, repo_paths: List[Path]):
        """Main analysis pipeline"""
        for repo_path in repo_paths:
            config = self._load_config(repo_path)
            for java_file in self._find_java_files(repo_path):
                analysis = self._analyze_file_with_llm(java_file, config, repo_path.name)
                self._add_to_graph(analysis, repo_path.name)

        self._generate_embeddings()
        self._save_graph()

    def _load_config(self, repo_path: Path) -> Dict:
        """Load application config"""
        for config_file in [repo_path/"src/main/resources/application.yml", 
                          repo_path/"src/main/resources/application.properties"]:
            if config_file.exists():
                return self._parse_config(config_file)
        return {}

    def _analyze_file_with_llm(self, file_path: Path, config: Dict, repo_name: str) -> Dict:
        """Get LLM analysis with foolproof JSON output"""
        prompt = self._build_llm_prompt(file_path, config, repo_name)
        response = self._call_llm_api(prompt)
        return self._validate_llm_response(response)

    def _build_llm_prompt(self, file_path: Path, config: Dict, repo_name: str) -> str:
        """Construct a bulletproof prompt for dependency analysis"""
        with open(file_path, 'r') as f:
            code = f.read()
        
        return f"""
        Analyze this Java file and generate a JSON response with:
        1. Class dependencies
        2. Responsibility description
        3. API/database interactions
        
        Required JSON format:
        {{
          "class_name": "...",
          "package": "...",
          "repository": "{repo_name}",
          "responsibility": "1-2 sentence description",
          "dependencies": {{
            "intra_repo": [{{"target": "Class", "type": "method_call/http/etc"}}],
            "inter_repo": [{{"target_repo": "...", "target_service": "...", "protocol": "..."}}],
            "external": [{{"system": "db/kafka", "details": "..."}}]
          }}
        }}
        
        Config:
        {json.dumps(config, indent=2)}
        
        Code:
        {code}
        
        Rules:
        - Return ONLY valid JSON
        - Include ALL dependencies
        - For HTTP calls, extract URLs from config
        """

    def _add_to_graph(self, analysis: Dict, repo_name: str):
        """Add analysis results to the graph"""
        node_id = f"{repo_name}:{analysis['package']}.{analysis['class_name']}"
        self.graph.add_node(node_id, **{
            'type': self._infer_node_type(analysis),
            'responsibility': analysis['responsibility'],
            'repo': repo_name
        })
        
        # Add dependencies as edges
        for dep in analysis['dependencies']['intra_repo']:
            self.graph.add_edge(node_id, f"{repo_name}:{dep['target']}", **dep)
        
        for dep in analysis['dependencies']['inter_repo']:
            self.graph.add_edge(node_id, f"{dep['target_repo']}:{dep['target_service']}", **dep)

    def _generate_embeddings(self):
        """Precompute embeddings for all nodes"""
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            text = f"{node} {node_data['type']} {node_data['responsibility']}"
            self.graph.nodes[node]['embedding'] = self.st_model.encode(text)

    def answer_question(self, question: str, technical: bool = False) -> str:
        """End-to-end question answering"""
        # 1. Semantic search
        query_embed = self.st_model.encode(question)
        similarities = {
            node: cosine_similarity(
                query_embed.reshape(1,-1), 
                self.graph.nodes[node]['embedding'].reshape(1,-1)
            )[0][0]
            for node in self.graph.nodes
        }
        top_nodes = sorted(similarities, key=similarities.get, reverse=True)[:3]
        
        # 2. Graph traversal
        context_nodes = set()
        for node in top_nodes:
            context_nodes.update(nx.descendants(self.graph, node))  # Downstream
            context_nodes.update(nx.ancestors(self.graph, node))    # Upstream
        
        # 3. Build LLM context
        context = self._build_llm_context(context_nodes)
        return self._generate_answer(question, context, technical)

    def _build_llm_context(self, nodes: set) -> str:
        """Convert graph segments to natural language"""
        context = []
        for node in nodes:
            data = self.graph.nodes[node]
            context.append(
                f"Component: {node}\n"
                f"Type: {data['type']}\n"
                f"Responsibility: {data['responsibility']}\n"
                f"Dependencies:\n" + 
                "\n".join(f"- {self.graph.edges[node, n]['type']} → {n}" 
                         for n in self.graph.successors(node))
            )
        return "\n\n".join(context)

# Usage Example
if __name__ == "__main__":
    analyzer = DependencyAnalyzer()
    repo_paths = analyzer.clone_repos(REPO_URLS)
    analyzer.analyze_codebase(repo_paths)
    
    while True:
        question = input("\nAsk a question (or 'quit'): ")
        if question.lower() == 'quit':
            break
        print(analyzer.answer_question(question, technical=True))