import networkx as nx
H = nx.DiGraph()

# add the attributes of nodes
H.add_nodes_from([
    (0, {"colour": "gray", "number":500}),
    (1, {"colour": "red", "number":1000}),
    (2, {"colour": "blue", "number":1000}),
    (3, {"colour": "black", "number":2000}),
    (4, {"colour": "pink", "number":1500})
])
# for node in H.nodes(data=True):
#   print(node)


# add edge attributes
H.add_edges_from([
  (0, 1),
  (1, 2),
  (2, 0),
  (2, 3),
  (3, 2),
  (3, 4)
])
# print(H.edges())

def print_graph_info(graph):
  print("Directed graph:", graph.is_directed())
  print("Number of nodes:", graph.number_of_nodes())
  print("Number of edges:", graph.number_of_edges())

# print_graph_info(H)


node_colours = list(nx.get_node_attributes(H, "colour").values())
print(node_colours)

node_sizes = list(nx.get_node_attributes(H, "number").values())
print(node_sizes)

nx.draw(H, with_labels = True, node_color = node_colours, node_size = node_sizes)