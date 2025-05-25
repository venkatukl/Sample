import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# 1. Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Load graph (with real embeddings)
with open('dependency_graph_real_embeddings.json') as f:
    graph = json.load(f)

# 3. Test query
query = "Where are bank accounts managed?"
query_embed = model.encode(query).reshape(1, -1)
print(query)
# 4. Compare against nodes
for node in graph['nodes']:
    if 'embedding' in node:
        node_embed = np.array(node['embedding']).reshape(1, -1)
        similarity = cosine_similarity(query_embed, node_embed)[0][0]
        print(f"{node['id']} ({node['type']}): {similarity:.3f} - {node['responsibility']}")