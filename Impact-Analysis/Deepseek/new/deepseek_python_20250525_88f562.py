class GraphQAEngine:
    def __init__(self, retriever: GraphAwareRetriever, llm_api):
        self.retriever = retriever
        self.llm_api = llm_api
    
    def answer(self, question: str, technical: bool = False) -> str:
        # Step 1: Classify question type
        question_type = self._classify_question(question)
        
        # Step 2: Retrieve relevant graph segments
        if question_type == "high_level":
            context = self._get_architecture_overview()
        elif question_type == "impact":
            context = self._analyze_impact(question)
        else:
            context = self._semantic_search_context(question)
        
        # Step 3: Generate LLM response
        return self._generate_response(question, context, technical)
    
    def _classify_question(self, question: str) -> str:
        """Determine question type using few-shot LLM classification"""
        prompt = f"""
        Classify this question about a codebase:
        
        Question: {question}
        
        Types:
        1. high_level - About overall architecture ("Explain the system")
        2. impact - About changes/dependencies ("What breaks if I modify X?")
        3. specific - About concrete implementation ("How does class Y work?")
        
        Respond ONLY with the type keyword.
        """
        return self.llm_api.generate(prompt, max_tokens=10).strip().lower()
    
    def _get_architecture_overview(self) -> str:
        """Generate high-level summary from graph structure"""
        repos = list(self.retriever.graph.repo_services.keys())
        summary = []
        
        for repo in repos:
            services = self.retriever.graph.repo_services[repo]
            summary.append(f"Repository {repo} contains {len(services)} services including {services[:3]}...")
            
            # Count inter-repo connections
            external_deps = set()
            for service in services:
                full_id = f"{repo}:{service}"
                for _, target, _ in self.retriever.graph.nx_graph.out_edges(full_id, data=True):
                    if target.split(":")[0] != repo:
                        external_deps.add(target.split(":")[0])
            
            if external_deps:
                summary.append(f"  - Connects to: {', '.join(external_deps)}")
        
        return "\n".join(summary)
    
    def _analyze_impact(self, question: str) -> str:
        """Use graph traversal to determine change impact"""
        # Extract target class from question using LLM
        prompt = f"""
        From this question, extract JUST the Java class name:
        Question: {question}
        
        Examples:
        - "What happens if we change UserService?" → UserService
        - "Impact of modifying return type in PaymentController?" → PaymentController
        
        Respond ONLY with the class name.
        """
        target_class = self.llm_api.generate(prompt, max_tokens=15).strip()
        
        # Find all occurrences across repos
        affected_nodes = []
        for node in self.retriever.graph.nx_graph.nodes():
            if node.endswith(target_class):
                affected_nodes.append(node)
        
        if not affected_nodes:
            return f"Class {target_class} not found in dependency graph"
        
        # Build impact report
        impact = []
        for node in affected_nodes:
            downstream = self.retriever.graph.get_downstream_impact(node)
            upstream = self.retriever.graph.get_upstream_dependencies(node)
            
            impact.append(f"## {node}\n")
            impact.append(f"**Directly impacts**: {len(downstream)} components")
            for dep in downstream[:3]:  # Show top 3
                impact.append(f"- {dep['class']} ({dep['relationship']})")
            
            if upstream:
                impact.append(f"\n**Depends on**: {len(upstream)} components")
                for dep in upstream[:3]:
                    impact.append(f"- {dep['class']} ({dep['relationship']})")
        
        return "\n".join(impact)
    
    def _generate_response(self, question: str, context: str, technical: bool) -> str:
        """Generate final answer with LLM"""
        prompt = f"""
        Answer this {'technical' if technical else 'non-technical'} question:
        
        Question: {question}
        
        Context:
        {context}
        
        Rules:
        - Be concise
        - Cite specific classes/repositories when possible
        - For technical answers, include dependency types (HTTP, Kafka, etc.)
        - For non-technical answers, focus on business capabilities
        """
        return self.llm_api.generate(prompt, max_tokens=500)