import streamlit as st
from sentence_transformers import SentenceTransformer
import requests
import json
from graph_utils import load_dependency_graph, get_relevant_graph_data
from embed_and_retrieve import create_embeddings, retrieve_relevant_files

def query_llm(prompt, context, graph_data, api_endpoint="https://your-llm-api-endpoint"):
    """
    Query the LLM API with code context and dependency graph data.
    
    Args:
        prompt (str): User query.
        context (str): Relevant code or documentation.
        graph_data (dict): Relevant nodes and edges from the dependency graph.
        api_endpoint (str): LLM API endpoint.
    Returns:
        str: LLM response.
    """
    graph_summary = f"Dependency Graph:\nNodes: {json.dumps(graph_data['nodes'], indent=2)}\nEdges: {json.dumps(graph_data['edges'], indent=2)}"
    full_prompt = f"Context:\n{context}\n\n{graph_summary}\n\nQuestion: {prompt}\nAnswer in detail, considering both high-level architecture and granular code-level impacts. For impact analysis, trace dependencies using the graph."
    
    payload = {
        'prompt': full_prompt,
        'max_tokens': 1000
    }
    try:
        response = requests.post(api_endpoint, json=payload)
        response.raise_for_status()
        return response.json().get('text', 'No response from LLM.')
    except Exception as e:
        return f"Error querying LLM: {e}"

def main():
    st.title("Codebase Query Interface with Dependency Graph")
    
    # Load embeddings and dependency graph
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index, file_mapping = create_embeddings(["repos/repo_0_repo1", "repos/repo_1_repo2", "repos/repo_2_repo3"])
    graph = load_dependency_graph()
    
    # Chat interface
    user_query = st.text_input("Ask a question about the codebase:")
    if user_query:
        # Retrieve relevant files and graph data
        relevant_files = retrieve_relevant_files(user_query, index, file_mapping, model, k=5)
        context = "\n\n".join([f"File: {file_path}\n{chunk}" for file_path, chunk in relevant_files])
        relevant_graph_data = get_relevant_graph_data(graph, user_query)
        
        # Query LLM
        response = query_llm(user_query, context, relevant_graph_data)
        st.write("**Response**:")
        st.write(response)

if __name__ == "__main__":
    main()