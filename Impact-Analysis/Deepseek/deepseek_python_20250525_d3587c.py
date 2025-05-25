import tiktoken  # For token counting
from typing import Optional

class CodeAnalyzer:
    def __init__(self, llm_api):
        self.llm_api = llm_api
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
    
    def analyze_file(self, file_path: Path, config_data: Dict, max_tokens: int = 128000) -> Dict:
        """
        Analyze a Java file with LLM, handling large files by chunking
        """
        with open(file_path, 'r') as f:
            code_content = f.read()
        
        # Get basic class info
        class_info = extract_class_info(file_path)
        
        # Prepare context with config
        context = {
            'config': config_data,
            'class_info': class_info,
            'repository': file_path.parts[1]  # Assumes repos are in a parent directory
        }
        
        # Prepare prompt
        prompt = f"""Analyze this Java class and its dependencies:
        
        Config:
        {json.dumps(context['config'], indent=2)}
        
        Class Info:
        Package: {context['class_info']['package']}
        Class: {context['class_info']['class_name']}
        Interface: {context['class_info']['is_interface']}
        
        Code:
        {code_content}
        
        Questions:
        1. What are the external dependencies (other classes, services, databases)?
        2. What APIs does this class expose (for controllers) or consume (for clients)?
        3. What database tables/MQ topics does it interact with?
        4. Are there any inter-repository dependencies?
        5. What is the primary responsibility of this class?
        
        Return as JSON with these keys:
        - class_name, package, file_path
        - external_dependencies (list of {class: "", package: "", type: "class/service/db/mq"})
        - exposed_apis (list of {name: "", method: "", path: "", return_type: ""})
        - consumed_apis (same structure as exposed_apis)
        - db_dependencies (list of {table: "", operation: ""})
        - mq_dependencies (list of {topic: "", operation: ""})
        - primary_responsibility
        - inter_repo_dependencies (list of {repo: "", service: ""})
        """
        
        # Check token count and chunk if necessary
        token_count = self.count_tokens(prompt)
        if token_count > max_tokens:
            return self._analyze_large_file(file_path, context, max_tokens)
        
        # Call LLM API
        response = self.llm_api.generate(
            prompt=prompt,
            max_output_tokens=4000,
            temperature=0.1
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print(f"Failed to parse LLM response for {file_path}")
            return None
    
    def _analyze_large_file(self, file_path: Path, context: Dict, max_tokens: int) -> Dict:
        """
        Handle large files by analyzing methods separately
        """
        # Implement method-level chunking here
        # This would involve:
        # 1. Splitting the file into individual methods
        # 2. Analyzing each method separately
        # 3. Combining the results
        pass