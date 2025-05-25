def build_analysis_prompt(file_content: str, config: dict) -> str:
    return f"""
    Analyze this Java class and provide:
    1. Dependencies (as previously requested)
    2. A concise responsibility statement
    
    Output JSON format:
    {{
      "class_name": "...",
      "package": "...",
      "dependencies": {{...}},
      "responsibility": "What this class does in 1-2 sentences"
    }}
    
    Code to analyze:
    {file_content}
    
    Config context:
    {json.dumps(config)}
    """