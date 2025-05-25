def build_full_dependency_graph(repo_urls):
    graph = ConsolidatedGraph()
    
    for repo_url in repo_urls:
        repo_path = clone_repository(repo_url)
        config = load_config(repo_path)
        
        for java_file in find_java_files(repo_path):
            prompt = build_llm_prompt(
                file_content=java_file.read_text(),
                config=config,
                repo_name=repo_path.name
            )
            
            analysis = get_valid_llm_response(prompt)
            if analysis:
                analysis["repository"] = repo_path.name  # Tag with repo
                graph.add_dependencies(analysis)
    
    # Save final graph
    with open("dependency_graph.json", "w") as f:
        json.dump(graph.to_json(), f, indent=2)