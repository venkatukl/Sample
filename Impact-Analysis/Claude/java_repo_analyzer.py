#!/usr/bin/env python3
"""
Comprehensive Java Repository Analysis System
Analyzes multiple Java Spring repositories and provides intelligent Q&A capabilities
"""

import os
import json
import re
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess
import logging
from collections import defaultdict

import git
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodeElement:
    """Represents a code element (class, method, API endpoint, etc.)"""
    type: str  # 'class', 'method', 'api_endpoint', 'database_entity', etc.
    name: str
    file_path: str
    line_number: int
    repository: str
    content: str
    dependencies: List[str]
    metadata: Dict[str, Any]

@dataclass
class Dependency:
    """Represents a dependency relationship"""
    source: str
    target: str
    type: str  # 'api_call', 'database_access', 'message_queue', 'import'
    repository_source: str
    repository_target: str
    metadata: Dict[str, Any]

class JavaPatternExtractor:
    """Extracts Java patterns without AST parsing"""
    
    def __init__(self):
        self.patterns = {
            'class_declaration': r'(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w\s,]+))?',
            'interface_declaration': r'(?:public|private|protected)?\s*interface\s+(\w+)(?:\s+extends\s+([\w\s,]+))?',
            'method_declaration': r'(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?(?:abstract\s+)?(\w+(?:<[^>]+>)?)\s+(\w+)\s*\([^)]*\)',
            'spring_annotations': r'@(RestController|Controller|Service|Repository|Component|Entity|Table|RequestMapping|GetMapping|PostMapping|PutMapping|DeleteMapping|Autowired|Value|ConfigurationProperties)',
            'api_endpoints': r'@(GetMapping|PostMapping|PutMapping|DeleteMapping|RequestMapping)\s*(?:\([^)]*\))?',
            'database_annotations': r'@(Entity|Table|Column|Id|GeneratedValue|OneToMany|ManyToOne|ManyToMany|OneToOne)',
            'imports': r'import\s+((?:static\s+)?[a-zA-Z_][\w.]*(?:\.\*)?);',
            'package': r'package\s+([a-zA-Z_][\w.]*);',
            'method_calls': r'(\w+)\.(\w+)\s*\(',
            'sql_queries': r'(?:@Query\s*\(\s*["\']([^"\']+)["\']|createQuery\s*\(\s*["\']([^"\']+)["\'])',
            'message_queue': r'@(JmsListener|RabbitListener|KafkaListener|EventListener)',
            'configuration': r'@(Configuration|ConfigurationProperties|Value)\s*(?:\([^)]*\))?'
        }
    
    def extract_elements(self, content: str, file_path: str, repository: str) -> List[CodeElement]:
        """Extract code elements from Java source"""
        elements = []
        lines = content.split('\n')
        
        # Extract package
        package_match = re.search(self.patterns['package'], content)
        package_name = package_match.group(1) if package_match else ""
        
        # Extract classes
        for match in re.finditer(self.patterns['class_declaration'], content, re.MULTILINE):
            class_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            
            elements.append(CodeElement(
                type='class',
                name=f"{package_name}.{class_name}" if package_name else class_name,
                file_path=file_path,
                line_number=line_num,
                repository=repository,
                content=self._extract_class_content(content, match.start()),
                dependencies=self._extract_class_dependencies(content, match.start()),
                metadata={'package': package_name, 'extends': match.group(2), 'implements': match.group(3)}
            ))
        
        # Extract API endpoints
        for match in re.finditer(self.patterns['api_endpoints'], content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            endpoint_content = self._extract_method_around_line(content, line_num)
            
            elements.append(CodeElement(
                type='api_endpoint',
                name=self._extract_endpoint_name(endpoint_content),
                file_path=file_path,
                line_number=line_num,
                repository=repository,
                content=endpoint_content,
                dependencies=self._extract_method_dependencies(endpoint_content),
                metadata={'annotation': match.group(1), 'http_method': match.group(1).replace('Mapping', '')}
            ))
        
        return elements
    
    def _extract_class_content(self, content: str, start_pos: int) -> str:
        """Extract full class content"""
        lines = content[start_pos:].split('\n')
        brace_count = 0
        class_lines = []
        
        for line in lines:
            class_lines.append(line)
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0 and '{' in line:
                break
        
        return '\n'.join(class_lines[:50])  # Limit to prevent token overflow
    
    def _extract_class_dependencies(self, content: str, start_pos: int) -> List[str]:
        """Extract dependencies for a class"""
        dependencies = []
        
        # Extract imports
        for match in re.finditer(self.patterns['imports'], content):
            dependencies.append(match.group(1))
        
        # Extract method calls
        class_content = self._extract_class_content(content, start_pos)
        for match in re.finditer(self.patterns['method_calls'], class_content):
            dependencies.append(f"{match.group(1)}.{match.group(2)}")
        
        return list(set(dependencies))
    
    def _extract_method_around_line(self, content: str, line_num: int) -> str:
        """Extract method content around a specific line"""
        lines = content.split('\n')
        start = max(0, line_num - 10)
        end = min(len(lines), line_num + 20)
        return '\n'.join(lines[start:end])
    
    def _extract_endpoint_name(self, content: str) -> str:
        """Extract API endpoint name"""
        method_match = re.search(r'public\s+\w+\s+(\w+)\s*\(', content)
        return method_match.group(1) if method_match else "unknown_endpoint"
    
    def _extract_method_dependencies(self, content: str) -> List[str]:
        """Extract dependencies for a method"""
        dependencies = []
        
        # Method calls
        for match in re.finditer(self.patterns['method_calls'], content):
            dependencies.append(f"{match.group(1)}.{match.group(2)}")
        
        # Database queries
        for match in re.finditer(self.patterns['sql_queries'], content):
            query = match.group(1) or match.group(2)
            if query:
                dependencies.append(f"sql:{query[:100]}")
        
        return dependencies

class LLMAnalyzer:
    """LLM-based code analysis"""
    
    def __init__(self, llm_api_endpoint: str, api_key: str):
        self.llm_api_endpoint = llm_api_endpoint
        self.api_key = api_key
        self.max_tokens = 8000  # Conservative limit for Gemini Flash 2.0
    
    async def analyze_code_chunk(self, code_chunk: str, repository: str) -> Dict[str, Any]:
        """Analyze code chunk using LLM"""
        prompt = f"""
        Analyze this Java Spring Framework code from repository '{repository}':

        {code_chunk}

        Provide analysis in JSON format with:
        1. summary: Brief overview of the code's purpose
        2. components: List of main classes/interfaces with their roles
        3. api_endpoints: List of REST endpoints with HTTP methods and paths
        4. dependencies: External dependencies and services used
        5. database_entities: Database tables/entities referenced
        6. business_logic: Key business operations performed
        7. integration_points: External systems or APIs called
        8. potential_issues: Code quality or architectural concerns

        Respond with valid JSON only.
        """
        
        try:
            response = await self._call_llm(prompt)
            return json.loads(response)
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {"error": str(e), "summary": "Analysis failed"}
    
    async def generate_dependency_relationships(self, elements: List[CodeElement]) -> List[Dependency]:
        """Generate dependencies using LLM analysis"""
        dependencies = []
        
        # Process in chunks to avoid token limits
        chunk_size = 5
        for i in range(0, len(elements), chunk_size):
            chunk = elements[i:i+chunk_size]
            chunk_data = [{"name": e.name, "type": e.type, "content": e.content[:500], "repository": e.repository} for e in chunk]
            
            prompt = f"""
            Analyze these code elements and identify dependencies between them:

            {json.dumps(chunk_data, indent=2)}

            For each dependency relationship found, provide JSON with:
            - source: Source element name
            - target: Target element name  
            - type: Relationship type (api_call, database_access, service_injection, etc.)
            - confidence: Confidence level (0.0-1.0)
            - description: Brief description of the relationship

            Respond with JSON array of dependency objects.
            """
            
            try:
                response = await self._call_llm(prompt)
                llm_deps = json.loads(response)
                
                for dep in llm_deps:
                    dependencies.append(Dependency(
                        source=dep['source'],
                        target=dep['target'],
                        type=dep['type'],
                        repository_source=self._find_repository(dep['source'], elements),
                        repository_target=self._find_repository(dep['target'], elements),
                        metadata={'confidence': dep.get('confidence', 0.5), 'description': dep.get('description', '')}
                    ))
            except Exception as e:
                logger.error(f"Dependency analysis failed: {e}")
        
        return dependencies
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM API"""
        payload = {
            "model": "gemini-flash-2.0",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0.1
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(self.llm_api_endpoint, json=payload, headers=headers)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
    
    def _find_repository(self, element_name: str, elements: List[CodeElement]) -> str:
        """Find repository for an element"""
        for element in elements:
            if element.name == element_name:
                return element.repository
        return "unknown"

class VectorStore:
    """Vector storage and retrieval using ChromaDB"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="code_analysis",
            metadata={"hnsw:space": "cosine"}
        )
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_elements(self, elements: List[CodeElement]):
        """Add code elements to vector store"""
        documents = []
        metadatas = []
        ids = []
        
        for i, element in enumerate(elements):
            # Create searchable document
            doc_content = f"""
            Type: {element.type}
            Name: {element.name}
            Repository: {element.repository}
            File: {element.file_path}
            Content: {element.content}
            Dependencies: {', '.join(element.dependencies)}
            Metadata: {json.dumps(element.metadata)}
            """
            
            documents.append(doc_content)
            metadatas.append({
                "type": element.type,
                "name": element.name,
                "repository": element.repository,
                "file_path": element.file_path,
                "line_number": element.line_number
            })
            ids.append(f"{element.repository}_{element.type}_{i}")
        
        # Split large batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            self.collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
    
    def search(self, query: str, n_results: int = 10, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search for relevant code elements"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_dict
        )
        
        return [
            {
                "content": doc,
                "metadata": meta,
                "distance": dist
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0], 
                results["distances"][0]
            )
        ]

class RepositoryAnalyzer:
    """Main repository analysis orchestrator"""
    
    def __init__(self, llm_api_endpoint: str, api_key: str):
        self.pattern_extractor = JavaPatternExtractor()
        self.llm_analyzer = LLMAnalyzer(llm_api_endpoint, api_key)
        self.vector_store = VectorStore()
        self.repositories: Dict[str, str] = {}
        self.all_elements: List[CodeElement] = []
        self.dependency_graph: List[Dependency] = []
    
    async def analyze_repositories(self, repo_urls: List[str], clone_dir: str = "./repositories"):
        """Analyze multiple repositories"""
        os.makedirs(clone_dir, exist_ok=True)
        
        # Clone repositories
        for repo_url in repo_urls:
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            repo_path = os.path.join(clone_dir, repo_name)
            
            if not os.path.exists(repo_path):
                logger.info(f"Cloning {repo_url}")
                git.Repo.clone_from(repo_url, repo_path)
            
            self.repositories[repo_name] = repo_path
        
        # Process each repository
        for repo_name, repo_path in self.repositories.items():
            logger.info(f"Processing repository: {repo_name}")
            await self._process_repository(repo_name, repo_path)
        
        # Generate cross-repository dependencies
        logger.info("Generating dependency graph")
        self.dependency_graph = await self.llm_analyzer.generate_dependency_relationships(self.all_elements)
        
        # Index in vector store
        logger.info("Indexing in vector store")
        self.vector_store.add_elements(self.all_elements)
        
        # Save dependency graph
        self._save_dependency_graph()
    
    async def _process_repository(self, repo_name: str, repo_path: str):
        """Process a single repository"""
        java_files = list(Path(repo_path).rglob("*.java"))
        
        for java_file in java_files:
            try:
                with open(java_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract elements using pattern matching
                elements = self.pattern_extractor.extract_elements(
                    content, str(java_file), repo_name
                )
                self.all_elements.extend(elements)
                
                # LLM analysis for complex understanding
                if len(content) < 4000:  # Only analyze smaller files with LLM
                    llm_analysis = await self.llm_analyzer.analyze_code_chunk(content, repo_name)
                    
                    # Enhance elements with LLM insights
                    for element in elements:
                        if element.file_path == str(java_file):
                            element.metadata.update(llm_analysis)
                
            except Exception as e:
                logger.error(f"Error processing {java_file}: {e}")
    
    def _save_dependency_graph(self):
        """Save dependency graph as JSON"""
        graph_data = {
            "repositories": list(self.repositories.keys()),
            "elements": [asdict(element) for element in self.all_elements],
            "dependencies": [asdict(dep) for dep in self.dependency_graph],
            "metadata": {
                "total_elements": len(self.all_elements),
                "total_dependencies": len(self.dependency_graph),
                "analysis_timestamp": __import__('datetime').datetime.now().isoformat()
            }
        }
        
        with open("dependency_graph.json", "w") as f:
            json.dump(graph_data, f, indent=2)
        
        logger.info("Dependency graph saved to dependency_graph.json")

class ChatInterface:
    """Interactive chat interface for code Q&A"""
    
    def __init__(self, analyzer: RepositoryAnalyzer):
        self.analyzer = analyzer
        self.conversation_history = []
    
    async def ask_question(self, question: str) -> str:
        """Process user question and provide answer"""
        # Retrieve relevant context
        search_results = self.analyzer.vector_store.search(question, n_results=5)
        
        # Build context
        context_parts = []
        for result in search_results:
            context_parts.append(f"""
            Repository: {result['metadata']['repository']}
            Type: {result['metadata']['type']}
            Name: {result['metadata']['name']}
            Content: {result['content'][:1000]}
            """)
        
        context = "\n".join(context_parts)
        
        # Create comprehensive prompt
        prompt = f"""
        Based on the following code analysis from Java Spring repositories, answer the user's question.

        CONTEXT:
        {context}

        DEPENDENCY GRAPH INFO:
        Total repositories: {len(self.analyzer.repositories)}
        Total code elements: {len(self.analyzer.all_elements)}
        Total dependencies: {len(self.analyzer.dependency_graph)}

        USER QUESTION: {question}

        Provide a comprehensive answer that:
        1. Directly addresses the question
        2. References specific code elements when relevant
        3. Explains relationships and dependencies
        4. Considers impact across repositories
        5. Uses technical terms appropriately for the audience

        If the question is about impact analysis, trace through dependencies and explain potential effects.
        """
        
        try:
            response = await self.analyzer.llm_analyzer._call_llm(prompt)
            
            # Store in conversation history
            self.conversation_history.append({
                "question": question,
                "answer": response,
                "context_used": len(search_results)
            })
            
            return response
        
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"

# Example usage
async def main():
    """Example usage of the system"""
    
    # Configuration
    LLM_API_ENDPOINT = "https://your-llm-api.com/v1/chat/completions"
    API_KEY = "your-api-key"
    
    REPO_URLS = [
        "https://github.com/user/repo1.git",
        "https://github.com/user/repo2.git", 
        "https://github.com/user/repo3.git"
    ]
    
    # Initialize analyzer
    analyzer = RepositoryAnalyzer(LLM_API_ENDPOINT, API_KEY)
    
    # Analyze repositories
    await analyzer.analyze_repositories(REPO_URLS)
    
    # Initialize chat interface
    chat = ChatInterface(analyzer)
    
    # Example questions
    questions = [
        "Give me a high-level overview of all three applications",
        "What are the main API endpoints across all repositories?",
        "If I change the return type of UserService.getUser() method, what would be impacted?",
        "What database entities are shared between repositories?",
        "Explain the message queue integration patterns used"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        answer = await chat.ask_question(question)
        print(f"A: {answer}")

if __name__ == "__main__":
    asyncio.run(main())
