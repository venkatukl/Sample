# requirements.txt
gitpython==3.1.40
sentence-transformers==2.2.2
chromadb==0.4.18
numpy==1.24.3
requests==2.31.0
asyncio-throttle==1.0.2
pathlib2==2.3.7
python-dotenv==1.0.0

# Additional dependencies for enhanced functionality
javalang==0.13.0  # Optional: Java AST parsing if needed
tree-sitter==0.20.4  # Alternative parsing option
tree-sitter-java==0.20.2

# For web interface (optional)
fastapi==0.104.1
uvicorn==0.24.0
streamlit==1.28.1

"""
Installation and Setup Guide
===========================

1. Install Python 3.8+ and pip
2. Install requirements:
   pip install -r requirements.txt

3. Set environment variables:
   export LLM_API_ENDPOINT="your-llm-endpoint"
   export LLM_API_KEY="your-api-key"

4. Create .env file:
   LLM_API_ENDPOINT=https://your-api.com/v1/chat/completions
   LLM_API_KEY=your-api-key
   CHROMA_PERSIST_DIR=./chroma_db
   CLONE_DIR=./repositories

5. Run the analyzer:
   python java_repo_analyzer.py

Memory Requirements:
- Minimum 8GB RAM for processing 3 medium repositories
- 16GB+ recommended for large repositories
- SSD storage recommended for ChromaDB performance

Token Management Strategy:
- Code chunks limited to 4000 characters for LLM analysis
- Vector embeddings use 384-dimension MiniLM model
- Context window managed through chunking and relevance scoring

Rate Limiting:
- Implements exponential backoff for LLM API calls
- Batch processing to optimize API usage
- Configurable delays between requests
"""

# Configuration file - config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # LLM Configuration
    LLM_API_ENDPOINT = os.getenv("LLM_API_ENDPOINT")
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_MAX_TOKENS = 8000
    LLM_MAX_REQUESTS_PER_MINUTE = 60
    
    # Vector Store Configuration
    VECTOR_STORE_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MAX_EMBEDDING_TOKENS = 500
    
    # Repository Configuration
    CLONE_DIR = os.getenv("CLONE_DIR", "./repositories")
    MAX_FILE_SIZE_MB = 5
    SUPPORTED_EXTENSIONS = [".java"]
    
    # Analysis Configuration
    CHUNK_SIZE = 4000
    OVERLAP_SIZE = 200
    MIN_CODE_CONFIDENCE = 0.7
    
    # Chat Configuration
    MAX_CONTEXT_RESULTS = 10
    CONVERSATION_HISTORY_LIMIT = 50

# Enhanced analyzer with configuration
class ConfigurableRepositoryAnalyzer(RepositoryAnalyzer):
    def __init__(self, config: Config):
        super().__init__(config.LLM_API_ENDPOINT, config.LLM_API_KEY)
        self.config = config
        self.rate_limiter = self._init_rate_limiter()
    
    def _init_rate_limiter(self):
        """Initialize rate limiter for API calls"""
        from asyncio_throttle import Throttler
        return Throttler(rate_limit=self.config.LLM_MAX_REQUESTS_PER_MINUTE, period=60)
    
    async def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Call LLM with retry logic and rate limiting"""
        async with self.rate_limiter:
            for attempt in range(max_retries):
                try:
                    await asyncio.sleep(attempt * 2)  # Exponential backoff
                    return await self.llm_analyzer._call_llm(prompt)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")

# Advanced dependency graph generator
class AdvancedDependencyAnalyzer:
    """Enhanced dependency analysis with multiple strategies"""
    
    def __init__(self, config: Config):
        self.config = config
        self.dependency_types = {
            'api_call': {'weight': 1.0, 'critical': True},
            'database_access': {'weight': 0.9, 'critical': True},
            'service_injection': {'weight': 0.8, 'critical': True},
            'message_queue': {'weight': 0.7, 'critical': True},
            'configuration': {'weight': 0.6, 'critical': False},
            'utility_call': {'weight': 0.4, 'critical': False}
        }
    
    def analyze_impact(self, change_target: str, dependency_graph: List[Dependency]) -> Dict[str, Any]:
        """Analyze impact of changing a specific component"""
        upstream_deps = [d for d in dependency_graph if d.target == change_target]
        downstream_deps = [d for d in dependency_graph if d.source == change_target]
        
        impact_analysis = {
            'direct_upstream': len(upstream_deps),
            'direct_downstream': len(downstream_deps),
            'critical_dependencies': [],
            'affected_repositories': set(),
            'risk_level': 'low'
        }
        
        # Calculate risk level
        critical_count = sum(1 for d in upstream_deps + downstream_deps 
                           if self.dependency_types.get(d.type, {}).get('critical', False))
        
        if critical_count > 5:
            impact_analysis['risk_level'] = 'high'
        elif critical_count > 2:
            impact_analysis['risk_level'] = 'medium'
        
        # Collect affected repositories
        for dep in upstream_deps + downstream_deps:
            impact_analysis['affected_repositories'].add(dep.repository_source)
            impact_analysis['affected_repositories'].add(dep.repository_target)
        
        impact_analysis['affected_repositories'] = list(impact_analysis['affected_repositories'])
        
        return impact_analysis

# Web interface using FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Java Repository Analyzer API")

class QuestionRequest(BaseModel):
    question: str
    repository_filter: Optional[str] = None
    max_results: Optional[int] = 10

class AnalysisRequest(BaseModel):
    repositories: List[str]
    clone_fresh: Optional[bool] = False

@app.post("/analyze")
async def analyze_repositories(request: AnalysisRequest):
    """Analyze repositories endpoint"""
    try:
        config = Config()
        analyzer = ConfigurableRepositoryAnalyzer(config)
        await analyzer.analyze_repositories(request.repositories)
        
        return {
            "status": "success",
            "message": f"Analyzed {len(request.repositories)} repositories",
            "elements_found": len(analyzer.all_elements),
            "dependencies_found": len(analyzer.dependency_graph)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask question endpoint"""
    try:
        # Load existing analyzer (would need to persist state)
        config = Config()
        analyzer = ConfigurableRepositoryAnalyzer(config)
        chat = ChatInterface(analyzer)
        
        answer = await chat.ask_question(request.question)
        
        return {
            "question": request.question,
            "answer": answer,
            "context_used": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dependency-graph")
async def get_dependency_graph():
    """Get dependency graph endpoint"""
    try:
        with open("dependency_graph.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dependency graph not found. Run analysis first.")

# Streamlit interface for non-technical users
import streamlit as st

def create_streamlit_interface():
    """Create Streamlit interface"""
    st.title("Java Repository Code Assistant")
    st.sidebar.title("Configuration")
    
    # Repository input
    st.sidebar.