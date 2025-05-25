def generate_answer(question: str, context: str, llm_api, technical: bool) -> str:
    prompt = f"""
    Answer this {'technical' if technical else 'business'} question based on the system's dependency graph:
    
    Question: {question}
    
    Relevant System Context:
    {context}
    
    Guidelines:
    1. {"Use technical terms like HTTP/JPA when relevant" if technical else "Explain in non-technical terms"}
    2. Cite specific components when possible
    3. Highlight cross-repository dependencies
    """
    
    return llm_api.generate(prompt, max_tokens=500)