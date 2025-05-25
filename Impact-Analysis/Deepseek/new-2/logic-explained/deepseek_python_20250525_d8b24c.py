# In your analysis loop
for java_file in java_files:
    analysis = llm_analyze_file(java_file)  # Gets both dependencies AND responsibility
    graph.add_node(
        id=f"{repo}:{analysis['package']}.{analysis['class_name']}",
        responsibility=analysis["responsibility"]  # Store with node
    )