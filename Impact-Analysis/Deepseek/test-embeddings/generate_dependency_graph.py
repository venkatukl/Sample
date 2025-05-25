import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'dependency_graph_sample.json')


# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
#print(model.cache_folder)

# Load the sample graph
with open(file_path) as f:
    graph = json.load(f)

# Generate Embeddings
for node in graph['nodes']:
    if 'responsibility' in node:
        # Create embedding text from node metadata
        text = f"{node['id']} {node['type']} {node['responsibility']}"
        node['embedding'] = model.encode(text).tolist()

# Save updated graph
with open('dependency_graph_real_embeddings.json', 'w') as f:
    json.dump(graph, f, indent=2)
