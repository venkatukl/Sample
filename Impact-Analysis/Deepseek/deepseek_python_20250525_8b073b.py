class CodeAssistant:
    def __init__(self, retriever: CodeRetriever, llm_api, graph: Dict):
        self.retriever = retriever
        self.llm_api = llm_api
        self.graph = graph
    
    def answer_question(self, question: str, technical_user: bool = False) -> str:
        """
        Answer questions about the codebase using retrieval and LLM
        """
        # Step 1: Retrieve relevant context
        search_results = self.retriever.search(question)
        
        # Step 2: Determine question type
        question_type = self._classify_question(question)
        
        # Step 3: Prepare context based on question type
        if question_type == 'high_level':
            context = self._get_high_level_context()
        elif question_type == 'dependency':
            context = self._get_dependency_context(question)
        else:
            context = '\n\n'.join([res['text'] for res in search_results[:3]])
        
        # Step 4: Generate answer with appropriate prompt
        if technical_user:
            prompt = self._technical_prompt(question, context)
        else:
            prompt = self._non_technical_prompt(question, context)
        
        return self.llm_api.generate(
            prompt=prompt,
            max_output_tokens=2000,
            temperature=0.3 if technical_user else 0.1
        )
    
    def _classify_question(self, question: str) -> str:
        """Classify question type to determine context strategy"""
        high_level_keywords = ['overview', 'high level', 'architecture', 'explain the system']
        dependency_keywords = ['dependency', 'impact', 'downstream', 'upstream', 'affect']
        
        question_lower = question.lower()
        
        if any(kw in question_lower for kw in high_level_keywords):
            return 'high_level'
        elif any(kw in question_lower for kw in dependency_keywords):
            return 'dependency'
        return 'granular'
    
    def _get_high_level_context(self) -> str:
        """Generate high-level context from dependency graph"""
        repos = ', '.join(self.graph['repositories'])
        num_services = len([n for n in self.graph['nodes'] if n['type'] != 'interface'])
        
        return f"""
        System Overview:
        - Repositories: {repos}
        - Total services: {num_services}
        - Key components:
          {self._extract_key_components()}
        """
    
    def _get_dependency_context(self, question: str) -> str:
        """Extract relevant dependency information from graph"""
        # This would analyze the question to find specific components
        # and return their dependencies from the graph
        return "Dependency context extracted from graph"
    
    def _technical_prompt(self, question: str, context: str) -> str:
        """Create prompt for technical users"""
        return f"""
        You are a senior software engineer answering technical questions about a Java/Spring codebase.
        Use the following context to answer the question precisely and technically.
        Include code references when appropriate.
        
        Question: {question}
        
        Context:
        {context}
        
        Answer concisely with technical accuracy:
        """
    
    def _non_technical_prompt(self, question: str, context: str) -> str:
        """Create prompt for non-technical users"""
        return f"""
        Explain the following in simple, non-technical terms for a business stakeholder:
        
        Question: {question}
        
        Context:
        {context}
        
        Provide a clear, jargon-free answer focusing on business impact:
        """