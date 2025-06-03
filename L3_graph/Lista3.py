import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from networkx.algorithms.community import girvan_newman, label_propagation_communities
from networkx.algorithms.community.quality import modularity
from scipy.io import mmread
'''
file_path = "ca-HepTh.mtx"
matrix = mmread(file_path)
G = nx.Graph(matrix)
'''
n_nodes = 8638
n_edges = 24827
'''
# Generowanie grafów losowych
best_m = 1
best_diff = float('inf')
for m_try in range(1, n_nodes):
    est_edges = m_try * (n_nodes - m_try)
    diff = abs(est_edges - n_edges)
    if diff < best_diff:
        best_diff = diff
        best_m = m_try
    if est_edges > n_edges:
        break
avg_degree = 2 * best_m

G_ba = nx.barabasi_albert_graph(n=n_nodes, m=max(1, avg_degree // 2), seed=42)
'''
G_er = nx.gnm_random_graph(n=n_nodes, m=n_edges, seed=42)

if not nx.is_connected(G_er):
    print("Graf G_er jest niespójny — wyodrębniam największą spójną składową...")
    largest_cc = max(nx.connected_components(G_er), key=len)
    G_er = G_er.subgraph(largest_cc).copy()
else:
    print("Graf G_er jest spójny.")

#Girvan-Newman
def girvan_newman_communities(G, level=1):
    comp = girvan_newman(G)
    for _ in range(level - 1):
        next(comp)
    return list(next(comp))

#Louvain Method
def louvain_communities(G):
    import community as community_louvain
    partition = community_louvain.best_partition(G)
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)
    return list(communities.values())

#Label Propagation
def label_propagation(G):
    return list(label_propagation_communities(G))

#Analiza jakości grupowań
def analyze_communities(G, communities):
    clustering = nx.average_clustering(G)
    mod = modularity(G, communities)
    sizes = sorted([len(c) for c in communities], reverse=True)
    return {
        "number_of_communities": len(communities),
        "average_clustering": clustering,
        "modularity": mod,
        "group_sizes": sizes[:10]}

#Rozkład wielkości grup
def group_size_distribution (communities):
        sizes = [len(group) for group in communities]
        plt.figure(figsize=(8, 5))
        plt.hist(sizes, bins=20, color='skyblue', edgecolor='black')
        plt.title('Rozkład wielkości społeczności')
        plt.xlabel('Liczba węzłów w społeczności')
        plt.ylabel('Liczba społeczności')
        plt.show()

#Wizualizacja
def draw_communities(G, communities, title):
    pos = nx.spring_layout(G, seed=42)
    color_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            color_map[node] = i
    node_colors = [color_map.get(node, 0) for node in G.nodes()]
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.colormaps['tab20'], node_size=40)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title(title)
    plt.axis('off')
    plt.show()

#communities_gn = girvan_newman_communities(G, level=1)
communities_lm = louvain_communities(G_er)
#communities_lp = label_propagation(G)

#print("Girvan-Newman:", analyze_communities(G, communities_gn))
print("Louvain:", analyze_communities(G_er, communities_lm))
#print("Label Propagation:", analyze_communities(G, communities_lp))

#group_size_distribution (communities_gn)
group_size_distribution (communities_lm)
#group_size_distribution (communities_lp)

#Wizualizacje
#draw_communities(G, communities_gn, "Girvan-Newman")
draw_communities(G_er, communities_lm, "Louvain")
#draw_communities(G, communities_lp, "Label Propagation")
'''
#Wybór największej składowej spójnej
G_sub = G.subgraph(max(nx.connected_components(G), key=len))

communities_gn = girvan_newman_communities(G_sub, level=1)
communities_lm = louvain_communities(G_sub)
communities_lp = label_propagation(G_sub)

print("Girvan-Newman:", analyze_communities(G_sub, communities_gn))
print("Louvain:", analyze_communities(G_sub, communities_lm))
print("Label Propagation:", analyze_communities(G_sub, communities_lp))

group_size_distribution (communities_gn)
group_size_distribution (communities_lm)
group_size_distribution (communities_lp)

#Wizualizacje
draw_communities(G_sub, communities_gn, "Girvan-Newman")
draw_communities(G_sub, communities_lm, "Louvain")
draw_communities(G_sub, communities_lp, "Label Propagation")
'''