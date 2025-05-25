def build_llm_prompt(file_content: str, config: dict, repo_name: str) -> str:
    return f"""
    Analyze the Java class below and extract ALL dependencies. 
    Return JSON matching this EXACT structure:
    
    {json.dumps(DEPENDENCY_SCHEMA, indent=2)}
    
    Example Output:
    {{
      "class_name": "PaymentService",
      "package": "com.example.payment",
      "repository": "repo1",
      "dependencies": {{
        "intra_repo": [
          {{"target_class": "PaymentRepository", "type": "method_call"}}
        ],
        "inter_repo": [
          {{"target_repo": "repo2", "target_service": "UserService", "protocol": "HTTP"}}
        ]
      }}
    }}
    
    Rules:
    1. Include ALL @Autowired fields, method calls to other classes, and inherited types
    2. For HTTP calls, look for RestTemplate/WebClient usage and @FeignClient interfaces
    3. For inter-repo calls, check application.yml for service URLs
    
    Config Context:
    {json.dumps(config, indent=2)}
    
    Code to Analyze:
    {file_content}
    """