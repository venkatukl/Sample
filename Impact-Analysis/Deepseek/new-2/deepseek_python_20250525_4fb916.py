# Initialize
index = GraphEmbeddingIndex('dependency_graph.json')

# User query
query = "Which services process payments?"
results = index.find_similar_nodes(query)

# Output example
[
    {
        "node_id": "payment-service:com.payment.PaymentService",
        "similarity": 0.87,
        "type": "Service",
        "repo": "payment-service",
        "responsibility": "Handles payment processing logic"
    },
    {
        "node_id": "payment-service:com.payment.PaymentController",
        "similarity": 0.76,
        "type": "RestController",
        "repo": "payment-service"
    }
]