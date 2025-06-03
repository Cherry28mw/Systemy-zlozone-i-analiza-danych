import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from sklearn.manifold import TSNE
import numpy as np
from scipy.io import mmread

file_path = "ca-HepTh.mtx"
matrix = mmread(file_path)
G = nx.Graph(matrix)

G_sub = G.subgraph(max(nx.connected_components(G), key=len)).copy()

#Centralność closeness
cc_full = nx.closeness_centrality(G)
color_full = [cc_full.get(node, 0) for node in G.nodes()]

closeness_centrality = nx.closeness_centrality(G_sub)
node_color_values = [closeness_centrality[node] for node in G_sub.nodes()]

#Spectral layout(subgraf)
plt.figure(figsize=(10, 8))
pos_spec = nx.spectral_layout(G_sub)
nx.draw(G_sub, pos=pos_spec, node_size=10, node_color=node_color_values, cmap=plt.cm.plasma)
plt.title("Spectral Layout - closeness centrality")
plt.savefig("Spectral_layout_sb.png", dpi=300)
plt.show()

#Shell_layout(subgraf)
plt.figure(figsize=(10, 8))
pos_shell_sub = nx.shell_layout(G_sub)
sizes = [300 * closeness_centrality[node] for node in G_sub.nodes()]
nodes = nx.draw_networkx_nodes(G_sub, pos=pos_shell_sub,node_size=sizes,node_color=node_color_values,cmap=plt.cm.plasma)
nx.draw_networkx_edges(G_sub, pos=pos_shell_sub, alpha=0.2)
plt.title("Shell Layout (spójny podgraf) - kolor: closeness centrality")
plt.colorbar(nodes, label="closeness centrality")
plt.axis('off')
plt.savefig("Shell_layout_sb.png", dpi=300)
plt.show()

#Shell_layout
plt.figure(figsize=(10, 8))
pos_shell_full = nx.shell_layout(G)
sizes = [300 * cc_full.get(node, 0) for node in G.nodes()]
nodes = nx.draw_networkx_nodes(G, pos=pos_shell_full, node_size=sizes, node_color=color_full, cmap=plt.cm.plasma)
nx.draw_networkx_edges(G, pos=pos_shell_full, alpha=0.05)
plt.title("Shell Layout (pełny graf) - kolor: closeness centrality")
plt.colorbar(nodes, label="closeness centrality")
plt.axis('off')
plt.savefig("Shell_layout.png", dpi=300)
plt.show()

#Node2Vec + t-SNE(wizualizacja)
node2vec = Node2Vec(G_sub, dimensions=64, walk_length=30, num_walks=200, workers=2, quiet=True)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

#Embedding
embeddings = np.array([model.wv[str(node)] for node in G_sub.nodes()])
labels = list(G_sub.nodes())

#t-SNE(wizualizacja)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings)

#Wizualizacja Node2Vec + t-SNE
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                      c=node_color_values, cmap=plt.cm.plasma, s=10)
plt.colorbar(scatter)
plt.title("Node2Vec + t-SNE - kolor: closeness centrality")
plt.savefig("Embedding.png", dpi=300)
plt.show()
