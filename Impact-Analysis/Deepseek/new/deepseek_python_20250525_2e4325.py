def analyze_file_with_llm(file_path: Path, config_data: Dict) -> Dict:
    # Step 1: Lightweight extraction (no full parsing)
    class_info = {
        "package": extract_package(file_path),
        "class_name": extract_class_name(file_path),
        "annotations": extract_spring_annotations(file_path)
    }
    
    # Step 2: Send critical metadata + raw code to LLM
    prompt = f"""
    Analyze dependencies for this class:
    
    Metadata:
    {json.dumps(class_info, indent=2)}
    
    Config:
    {json.dumps(config_data, indent=2)}
    
    Code:
    {file_path.read_text()}
    """
    
    # Step 3: LLM generates detailed dependencies
    return llm_api.generate(prompt)