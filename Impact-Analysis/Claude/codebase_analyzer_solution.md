# Multi-Repository Java Codebase Analyzer - Complete Solution

## Architecture Overview

The solution uses a hybrid approach combining traditional parsing with LLM-powered analysis to create a comprehensive understanding of your Java Spring repositories.

### Core Components:
1. **Repository Cloner & File Parser**
2. **LLM-Powered Dependency Analyzer**
3. **Multi-Level Embedding System**
4. **Intelligent Query Engine**
5. **Chat Interface**

## Phase 1: Repository Management & File Processing

### 1.1 Repository Cloning and File Discovery

```python
import os
import git
import json
from pathlib import Path
from typing import Dict, List, Set
import re
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class FileMetadata:
    repo_name: str
    file_path: str
    file_type: str
    package_name: str
    class_name: str
    methods: List[str]
    imports: List[str]
    annotations: List[str]
    size_bytes: int
    last_modified: str

class RepositoryManager:
    def __init__(self, base_dir: str = "./repos"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.repositories = {}
        
    def clone_repositories(self, repo_urls: List[str]) -> Dict[str, Path]:
        """Clone up to 3 repositories and return their local paths"""
        cloned_repos = {}
        
        for i, repo_url in enumerate(repo_urls[:3]):  # Limit to 3 repos
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            repo_path = self.base_dir / repo_name
            
            try:
                if repo_path.exists():
                    print(f"Repository {repo_name} already exists, pulling latest changes...")
                    repo = git.Repo(repo_path)
                    repo.remotes.origin.pull()
                else:
                    print(f"Cloning {repo_name}...")
                    git.Repo.clone_from(repo_url, repo_path)
                
                cloned_repos[repo_name] = repo_path
                self.repositories[repo_name] = repo_path
                
            except Exception as e:
                print(f"Failed to clone {repo_url}: {str(e)}")
                
        return cloned_repos
    
    def discover_java_files(self) -> Dict[str, List[Path]]:
        """Discover all Java files in cloned repositories"""
        java_files = {}
        
        for repo_name, repo_path in self.repositories.items():
            files = []
            for java_file in repo_path.rglob("*.java"):
                # Skip test files for now (can be included if needed)
                if not any(test_dir in str(java_file) for test_dir in ['test', 'tests']):
                    files.append(java_file)
            java_files[repo_name] = files
            
        return java_files
```

### 1.2 LLM-Powered File Parser

```python
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Any, Dict, List

class LLMFileParser:
    def __init__(self, llm_api_endpoint: str, api_key: str):
        self.llm_api_endpoint = llm_api_endpoint
        self.api_key = api_key
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def parse_java_file(self, file_path: Path, repo_name: str) -> FileMetadata:
        """Parse Java file using LLM to extract metadata"""
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Split large files into chunks to handle token limits
        chunks = self._split_content(content, max_size=3000)
        
        parsed_data = {
            'package_name': '',
            'class_name': '',
            'methods': [],
            'imports': [],
            'annotations': [],
            'api_endpoints': [],
            'database_entities': [],
            'dependencies': []
        }
        
        for chunk in chunks:
            chunk_data = self._parse_chunk_with_llm(chunk, file_path.name)
            self._merge_parsed_data(parsed_data, chunk_data)
        
        return FileMetadata(
            repo_name=repo_name,
            file_path=str(file_path.relative_to(Path('./repos'))),
            file_type=self._get_file_type(content),
            package_name=parsed_data['package_name'],
            class_name=parsed_data['class_name'],
            methods=parsed_data['methods'],
            imports=parsed_data['imports'],
            annotations=parsed_data['annotations'],
            size_bytes=len(content.encode('utf-8')),
            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        )
    
    def _split_content(self, content: str, max_size: int = 3000) -> List[str]:
        """Split content into manageable chunks"""
        if len(content) <= max_size:
            return [content]
        
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            if current_size + len(line) > max_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = len(line)
            else:
                current_chunk.append(line)
                current_size += len(line)
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _parse_chunk_with_llm(self, chunk: str, filename: str) -> Dict:
        """Use LLM to parse a code chunk"""
        
        prompt = f"""
        Analyze this Java code chunk from file '{filename}' and extract the following information in JSON format:
        
        1. package_name: The package declaration
        2. class_name: Main class/interface name
        3. methods: List of method names with their parameters
        4. imports: List of import statements
        5. annotations: List of Spring/Java annotations used
        6. api_endpoints: REST API endpoints if any (with HTTP methods)
        7. database_entities: JPA entities, repositories mentioned
        8. dependencies: Other classes/services referenced
        
        Code chunk:
        ```java
        {chunk}
        ```
        
        Respond with valid JSON only, no explanation:
        """
        
        try:
            response = self._call_llm_api(prompt)
            return json.loads(response)
        except Exception as e:
            print(f"Error parsing chunk: {str(e)}")
            return {
                'package_name': '',
                'class_name': '',
                'methods': [],
                'imports': [],
                'annotations': [],
                'api_endpoints': [],
                'database_entities': [],
                'dependencies': []
            }
    
    def _call_llm_api(self, prompt: str) -> str:
        """Call the in-house LLM API"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'gemini-flash-2.0',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 2000,
            'temperature': 0.1
        }
        
        response = requests.post(self.llm_api_endpoint, 
                               headers=headers, 
                               json=payload, 
                               timeout=30)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"LLM API error: {response.status_code}")
    
    def _merge_parsed_data(self, main_data: Dict, chunk_data: Dict):
        """Merge parsed data from multiple chunks"""
        for key, value in chunk_data.items():
            if isinstance(value, list):
                main_data[key].extend(value)
            elif isinstance(value, str) and value and not main_data[key]:
                main_data[key] = value
    
    def _get_file_type(self, content: str) -> str:
        """Determine the type of Java file"""
        if '@RestController' in content or '@Controller' in content:
            return 'controller'
        elif '@Service' in content:
            return 'service'
        elif '@Repository' in content:
            return 'repository'
        elif '@Entity' in content:
            return 'entity'
        elif '@Configuration' in content:
            return 'configuration'
        elif 'interface' in content:
            return 'interface'
        else:
            return 'class'
```

## Phase 2: Dependency Graph Generation

### 2.1 Comprehensive Dependency Analyzer

```python
@dataclass
class DependencyRelation:
    source_repo: str
    source_file: str
    source_class: str
    target_repo: str
    target_file: str
    target_class: str
    relation_type: str  # 'api_call', 'import', 'inheritance', 'composition', 'database'
    details: Dict[str, Any]

class DependencyGraphGenerator:
    def __init__(self, llm_parser: LLMFileParser):
        self.llm_parser = llm_parser
        self.file_metadata = {}
        self.dependencies = []
        
    def generate_comprehensive_graph(self, repo_files: Dict[str, List[Path]]) -> Dict:
        """Generate complete dependency graph"""
        
        # Step 1: Parse all files
        print("Parsing all files...")
        for repo_name, files in repo_files.items():
            self.file_metadata[repo_name] = []
            for file_path in files:
                try:
                    metadata = self.llm_parser.parse_java_file(file_path, repo_name)
                    self.file_metadata[repo_name].append(metadata)
                    print(f"Parsed: {file_path.name}")
                except Exception as e:
                    print(f"Error parsing {file_path}: {str(e)}")
        
        # Step 2: Analyze dependencies using LLM
        print("Analyzing dependencies...")
        self._analyze_dependencies()
        
        # Step 3: Generate final graph
        dependency_graph = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'repositories': list(repo_files.keys()),
                'total_files': sum(len(files) for files in self.file_metadata.values())
            },
            'repositories': {},
            'dependencies': [asdict(dep) for dep in self.dependencies]
        }
        
        # Add repository details
        for repo_name, files_metadata in self.file_metadata.items():
            dependency_graph['repositories'][repo_name] = {
                'files': [asdict(fm) for fm in files_metadata],
                'file_count': len(files_metadata),
                'packages': list(set(fm.package_name for fm in files_metadata if fm.package_name))
            }
        
        return dependency_graph
    
    def _analyze_dependencies(self):
        """Analyze dependencies between all files using LLM"""
        
        # Create a comprehensive context for LLM
        all_classes_context = self._build_classes_context()
        
        for repo_name, files_metadata in self.file_metadata.items():
            for file_meta in files_metadata:
                dependencies = self._find_file_dependencies(file_meta, all_classes_context)
                self.dependencies.extend(dependencies)
    
    def _build_classes_context(self) -> str:
        """Build context of all classes across repositories"""
        context_parts = []
        
        for repo_name, files_metadata in self.file_metadata.items():
            repo_classes = []
            for file_meta in files_metadata:
                if file_meta.class_name:
                    repo_classes.append(f"{file_meta.package_name}.{file_meta.class_name}")
            
            context_parts.append(f"Repository '{repo_name}' classes: {', '.join(repo_classes)}")
        
        return '\n'.join(context_parts)
    
    def _find_file_dependencies(self, file_meta: FileMetadata, context: str) -> List[DependencyRelation]:
        """Find dependencies for a specific file using LLM"""
        
        prompt = f"""
        Analyze the dependencies for this Java file and identify relationships with other classes/services.
        
        Available classes across all repositories:
        {context}
        
        File to analyze:
        - Repository: {file_meta.repo_name}
        - File: {file_meta.file_path}
        - Class: {file_meta.class_name}
        - Package: {file_meta.package_name}
        - Imports: {', '.join(file_meta.imports)}
        - Methods: {', '.join(file_meta.methods)}
        - Annotations: {', '.join(file_meta.annotations)}
        
        Identify dependencies and return as JSON array with this structure:
        [
          {{
            "target_class": "full.package.ClassName",
            "target_repo": "repository_name",
            "relation_type": "api_call|import|inheritance|composition|database",
            "details": {{
              "method_called": "method_name",
              "endpoint": "/api/path",
              "http_method": "GET|POST|PUT|DELETE",
              "description": "brief description"
            }}
          }}
        ]
        
        Only return valid JSON, no explanation:
        """
        
        try:
            response = self.llm_parser._call_llm_api(prompt)
            dependencies_data = json.loads(response)
            
            dependencies = []
            for dep_data in dependencies_data:
                dep = DependencyRelation(
                    source_repo=file_meta.repo_name,
                    source_file=file_meta.file_path,
                    source_class=f"{file_meta.package_name}.{file_meta.class_name}",
                    target_repo=dep_data.get('target_repo', 'unknown'),
                    target_file='',  # Will be resolved later
                    target_class=dep_data.get('target_class', ''),
                    relation_type=dep_data.get('relation_type', 'unknown'),
                    details=dep_data.get('details', {})
                )
                dependencies.append(dep)
            
            return dependencies
            
        except Exception as e:
            print(f"Error analyzing dependencies for {file_meta.class_name}: {str(e)}")
            return []
    
    def save_dependency_graph(self, graph: Dict, output_path: str = "./dependency_graph.json"):
        """Save the dependency graph to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(graph, f, indent=2)
        print(f"Dependency graph saved to {output_path}")
```

## Phase 3: Multi-Level Embedding System

### 3.1 Hierarchical Embedding Strategy

```python
from typing import Tuple
import pickle
import faiss

@dataclass
class CodeChunk:
    repo_name: str
    file_path: str
    chunk_type: str  # 'class', 'method', 'full_file', 'summary'
    content: str
    metadata: Dict
    embedding: np.ndarray = None

class MultiLevelEmbeddingSystem:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.chunks = []
        self.embeddings_index = None
        self.chunk_to_index = {}
        
    def create_embeddings(self, dependency_graph: Dict):
        """Create multi-level embeddings for all code"""
        
        print("Creating embeddings...")
        
        for repo_name, repo_data in dependency_graph['repositories'].items():
            for file_data in repo_data['files']:
                # Create different granularity chunks
                self._create_file_chunks(repo_name, file_data)
        
        # Create embeddings for all chunks
        self._generate_embeddings()
        
        # Build FAISS index for fast similarity search
        self._build_search_index()
        
        print(f"Created {len(self.chunks)} embeddings")
    
    def _create_file_chunks(self, repo_name: str, file_data: Dict):
        """Create multiple granularity chunks for a file"""
        
        # File-level summary chunk
        summary_content = f"""
        Repository: {repo_name}
        File: {file_data['file_path']}
        Type: {file_data['file_type']}
        Class: {file_data['class_name']}
        Package: {file_data['package_name']}
        Methods: {', '.join(file_data['methods'])}
        Annotations: {', '.join(file_data['annotations'])}
        """
        
        summary_chunk = CodeChunk(
            repo_name=repo_name,
            file_path=file_data['file_path'],
            chunk_type='summary',
            content=summary_content.strip(),
            metadata={
                'class_name': file_data['class_name'],
                'package_name': file_data['package_name'],
                'file_type': file_data['file_type'],
                'methods': file_data['methods']
            }
        )
        self.chunks.append(summary_chunk)
        
        # Method-level chunks (if we have detailed method info)
        for method in file_data['methods']:
            method_chunk = CodeChunk(
                repo_name=repo_name,
                file_path=file_data['file_path'],
                chunk_type='method',
                content=f"Method: {method} in class {file_data['class_name']}",
                metadata={
                    'method_name': method,
                    'class_name': file_data['class_name'],
                    'package_name': file_data['package_name']
                }
            )
            self.chunks.append(method_chunk)
    
    def _generate_embeddings(self):
        """Generate embeddings for all chunks"""
        contents = [chunk.content for chunk in self.chunks]
        
        # Process in batches to handle memory efficiently
        batch_size = 32
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i+batch_size]
            embeddings = self.embedding_model.encode(batch)
            
            for j, embedding in enumerate(embeddings):
                self.chunks[i + j].embedding = embedding
    
    def _build_search_index(self):
        """Build FAISS index for fast similarity search"""
        embeddings = np.array([chunk.embedding for chunk in self.chunks])
        
        # Use IndexFlatIP for cosine similarity
        self.embeddings_index = faiss.IndexFlatIP(embeddings.shape[1])
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.embeddings_index.add(embeddings)
        
        # Create mapping from index to chunk
        self.chunk_to_index = {i: chunk for i, chunk in enumerate(self.chunks)}
    
    def search_relevant_code(self, query: str, top_k: int = 10) -> List[Tuple[CodeChunk, float]]:
        """Search for relevant code chunks"""
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.embeddings_index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx in self.chunk_to_index:
                results.append((self.chunk_to_index[idx], float(score)))
        
        return results
    
    def save_embeddings(self, path: str = "./embeddings.pkl"):
        """Save embeddings to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'index': faiss.serialize_index(self.embeddings_index),
                'chunk_to_index': self.chunk_to_index
            }, f)
    
    def load_embeddings(self, path: str = "./embeddings.pkl"):
        """Load embeddings from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.embeddings_index = faiss.deserialize_index(data['index'])
            self.chunk_to_index = data['chunk_to_index']
```

## Phase 4: Intelligent Query Engine

### 4.1 Advanced Query Processing

```python
class IntelligentQueryEngine:
    def __init__(self, llm_parser: LLMFileParser, embedding_system: MultiLevelEmbeddingSystem, dependency_graph: Dict):
        self.llm_parser = llm_parser
        self.embedding_system = embedding_system
        self.dependency_graph = dependency_graph
        
    def process_query(self, query: str, user_type: str = "technical") -> Dict:
        """Process user query and generate comprehensive response"""
        
        # Step 1: Classify query type
        query_type = self._classify_query(query)
        
        # Step 2: Retrieve relevant context
        relevant_chunks = self.embedding_system.search_relevant_code(query, top_k=15)
        
        # Step 3: Build context for LLM
        context = self._build_query_context(query, relevant_chunks, query_type)
        
        # Step 4: Generate response using LLM
        response = self._generate_response(query, context, query_type, user_type)
        
        # Step 5: Add supporting information
        supporting_info = self._get_supporting_info(relevant_chunks, query_type)
        
        return {
            'query': query,
            'query_type': query_type,
            'response': response,
            'supporting_info': supporting_info,
            'relevant_files': [chunk.file_path for chunk, _ in relevant_chunks[:5]],
            'confidence_score': self._calculate_confidence(relevant_chunks)
        }
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of query"""
        
        classification_prompt = f"""
        Classify this query about a Java Spring codebase into one of these categories:
        
        1. "overview" - High-level application overview, architecture questions
        2. "impact_analysis" - Questions about changes, dependencies, impact
        3. "specific_code" - Questions about specific classes, methods, APIs
        4. "database" - Database-related questions
        5. "api" - REST API related questions
        6. "configuration" - Configuration and deployment questions
        7. "troubleshooting" - Debugging, error analysis
        
        Query: "{query}"
        
        Respond with just the category name:
        """
        
        try:
            return self.llm_parser._call_llm_api(classification_prompt).strip().lower()
        except:
            return "specific_code"
    
    def _build_query_context(self, query: str, relevant_chunks: List[Tuple[CodeChunk, float]], query_type: str) -> str:
        """Build comprehensive context for LLM"""
        
        context_parts = []
        
        # Add dependency graph summary
        context_parts.append("DEPENDENCY GRAPH SUMMARY:")
        context_parts.append(f"Total repositories: {len(self.dependency_graph['repositories'])}")
        
        for repo_name, repo_data in self.dependency_graph['repositories'].items():
            context_parts.append(f"- {repo_name}: {repo_data['file_count']} files")
        
        context_parts.append(f"Total dependencies: {len(self.dependency_graph['dependencies'])}")
        context_parts.append("")
        
        # Add relevant code chunks
        context_parts.append("RELEVANT CODE CONTEXT:")
        for i, (chunk, score) in enumerate(relevant_chunks[:10]):
            context_parts.append(f"[{i+1}] Repository: {chunk.repo_name}")
            context_parts.append(f"    File: {chunk.file_path}")
            context_parts.append(f"    Type: {chunk.chunk_type}")
            context_parts.append(f"    Content: {chunk.content[:500]}...")
            context_parts.append(f"    Relevance: {score:.3f}")
            context_parts.append("")
        
        # Add specific dependencies if impact analysis
        if query_type == "impact_analysis":
            context_parts.append("DEPENDENCY RELATIONSHIPS:")
            relevant_deps = self._find_relevant_dependencies(query, relevant_chunks)
            for dep in relevant_deps[:5]:
                context_parts.append(f"- {dep['source_class']} -> {dep['target_class']} ({dep['relation_type']})")
            context_parts.append("")
        
        return '\n'.join(context_parts)
    
    def _find_relevant_dependencies(self, query: str, relevant_chunks: List[Tuple[CodeChunk, float]]) -> List[Dict]:
        """Find dependencies relevant to the query"""
        relevant_files = {chunk.file_path for chunk, _ in relevant_chunks}
        relevant_deps = []
        
        for dep in self.dependency_graph['dependencies']:
            if dep['source_file'] in relevant_files or dep['target_file'] in relevant_files:
                relevant_deps.append(dep)
        
        return relevant_deps
    
    def _generate_response(self, query: str, context: str, query_type: str, user_type: str) -> str:
        """Generate response using LLM with full context"""
        
        system_prompt = self._get_system_prompt(query_type, user_type)
        
        user_prompt = f"""
        {system_prompt}
        
        CONTEXT:
        {context}
        
        USER QUERY: {query}
        
        Please provide a comprehensive answer based on the provided context.
        """
        
        return self.llm_parser._call_llm_api(user_prompt)
    
    def _get_system_prompt(self, query_type: str, user_type: str) -> str:
        """Get appropriate system prompt based on query and user type"""
        
        base_prompt = "You are an expert Java Spring Framework developer analyzing a multi-repository codebase."
        
        if user_type == "non-technical":
            base_prompt += " Explain things in simple terms, avoiding technical jargon where possible."
        
        type_specific = {
            "overview": "Focus on high-level architecture, main components, and overall system design.",
            "impact_analysis": "Analyze dependencies and provide detailed impact assessment of proposed changes.",
            "specific_code": "Provide detailed code-level explanations with examples.",
            "database": "Focus on data models, repositories, and database interactions.",
            "api": "Explain REST endpoints, request/response patterns, and API design.",
            "configuration": "Explain configuration files, properties, and deployment aspects.",
            "troubleshooting": "Help diagnose issues and provide debugging guidance."
        }
        
        return f"{base_prompt} {type_specific.get(query_type, '')}"
    
    def _get_supporting_info(self, relevant_chunks: List[Tuple[CodeChunk, float]], query_type: str) -> Dict:
        """Get additional supporting information"""
        
        repos_involved = list(set(chunk.repo_name for chunk, _ in relevant_chunks))
        file_types = list(set(chunk.metadata.get('file_type', 'unknown') for chunk, _ in relevant_chunks))
        
        return {
            'repositories_involved': repos_involved,
            'file_types_analyzed': file_types,
            'total_chunks_analyzed': len(relevant_chunks),
            'query_processing_time': datetime.now().isoformat()
        }
    
    def _calculate_confidence(self, relevant_chunks: List[Tuple[CodeChunk, float]]) -> float:
        """Calculate confidence score based on relevance scores"""
        if not relevant_chunks:
            return 0.0
        
        top_scores = [score for _, score in relevant_chunks[:3]]
        return sum(top_scores) / len(top_scores)
```

## Phase 5: Complete Integration & Chat Interface

### 5.1 Main Application Class

```python
import gradio as gr
from typing import Optional

class CodebaseAnalyzer:
    def __init__(self, llm_api_endpoint: str, api_key: str):
        self.repo_manager = RepositoryManager()
        self.llm_parser = LLMFileParser(llm_api_endpoint, api_key)
        self.dependency_generator = DependencyGraphGenerator(self.llm_parser)
        self.embedding_system = MultiLevelEmbeddingSystem(SentenceTransformer('all-MiniLM-L6-v2'))
        self.query_engine = None
        self.dependency_graph = None
        self.is_initialized = False
        
    def initialize_system(self, repo_urls: List[str]) -> str:
        """Initialize the complete system"""
        try:
            # Step 1: Clone repositories
            print("Step 1: Cloning repositories...")
            cloned_repos = self.repo_manager.clone_repositories(repo_urls)
            if not cloned_repos:
                return "Failed to clone any repositories"
            
            # Step 2: Discover Java files
            print("Step 2: Discovering Java files...")
            java_files = self.repo_manager.discover_java_files()
            total_files = sum(len(files) for files in java_files.values())
            print(f"Found {total_files} Java files across {len(java_files)} repositories")
            
            # Step 3: Generate dependency graph
            print("Step 3: Generating dependency graph...")
            self.dependency_graph = self.dependency_generator.generate_comprehensive_graph(java_files)
            self.dependency_generator.save_dependency_graph(self.dependency_graph)
            
            # Step 4: Create embeddings
            print("Step 4: Creating embeddings...")
            self.embedding_system.create_embeddings(self.dependency_graph)
            self.embedding_system.save_embeddings()
            
            # Step 5: Initialize query engine
            print("Step 5: Initializing query engine...")
            self.query_engine = IntelligentQueryEngine(
                self.llm_parser, 
                self.embedding_system, 
                self.dependency_graph
            )
            
            self.is_initialized = True
            
            return f"""
            ✅ System initialized successfully!
            
            📊 Analysis Summary:
            - Repositories: {len(cloned_repos)}
            - Total Java files: {total_files}
            - Dependencies found: {len(self.dependency_graph['dependencies'])}
            - Embeddings created: {len(self.embedding_system.chunks)}
            
            🚀 Ready to answer your questions!
            """
            
        except Exception as e:
            return f"❌ Initialization failed: {str(e)}"
    
    