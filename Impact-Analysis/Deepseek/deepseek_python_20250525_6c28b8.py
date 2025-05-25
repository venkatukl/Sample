def main(repo_urls):
    # 1. Clone repositories
    repo_paths = clone_repositories(repo_urls)
    
    # 2. Initialize components
    analyzer = CodeAnalyzer(llm_api=your_llm_api)
    graph_builder = DependencyGraphBuilder()
    retriever = CodeRetriever()
    
    # 3. Process each repository
    for repo_path in repo_paths:
        # Find all Java files
        java_files = find_java_files(repo_path)
        config_files = find_config_files(repo_path)
        config_data = parse_config_file(config_files.get('yaml') or config_files.get('properties'))
        
        # Analyze each file
        for java_file in java_files:
            # Analyze with LLM
            analysis = analyzer.analyze_file(java_file, config_data)
            
            if analysis:
                # Add to dependency graph
                graph_builder.add_analysis_result(analysis)
                
                # Add to retrieval system
                file_content = java_file.read_text()
                retriever.add_document(
                    text=file_content,
                    metadata={
                        'file_path': str(java_file),
                        'class_name': analysis['class_name'],
                        'package': analysis['package'],
                        'repository': repo_path.name
                    }
                )
    
    # 4. Save dependency graph
    graph_builder.save_graph(Path('dependency_graph.json'))
    
    # 5. Initialize QA system
    assistant = CodeAssistant(retriever, your_llm_api, graph_builder.graph)
    
    # (Here you would typically start a web service or CLI interface)
    print("System ready to answer questions")