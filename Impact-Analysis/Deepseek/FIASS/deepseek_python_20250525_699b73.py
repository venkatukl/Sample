# Load and query
import networkx as nx
G = nx.node_link_graph(json.load(open('dependency_graph.json'))
list(nx.descendants(G, "payment-service:com.payment.PaymentService"))