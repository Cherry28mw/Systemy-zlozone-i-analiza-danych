import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import mmread
'''
# Wczytanie grafu rzeczywistego
file_path = "ca-HepTh.mtx"
matrix = mmread(file_path)
G = nx.Graph(matrix)

# Znalezienie największej spójnej składowej
components = list(nx.connected_components(G))
largest_component = max(components, key=len)
G_sub = G.subgraph(largest_component).copy()

# Parametry największego spójnego podgrafu
n_nodes = G_sub.number_of_nodes()
n_edges = G_sub.number_of_edges()
avg_degree = (2 * n_edges) // n_nodes
'''
n_nodes = 8638
n_edges = 24827
# Generowanie grafów losowych
'''
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

# Wizualizacja największego spójnego podgrafu
degrees = dict(G_er.degree())
node_sizes = [degrees[n] * 2 for n in G_er.nodes]
edge_widths = [1 for _ in G_er.edges]

print("Obliczanie układu wierzchołków...")
pos = nx.spring_layout(G_er, seed=42)

plt.figure(figsize=(14, 14))
nx.draw_networkx_edges(G_er, pos, width=edge_widths, alpha=0.3)
nx.draw_networkx_nodes(G_er, pos, node_size=node_sizes, node_color='blue', alpha=0.6)
plt.title("Największy spójny podgraf ca-HepTh", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.savefig("graph_largest_component.png", dpi=300)
plt.show()

print("Wykres zapisany jako graph_largest_component.png")


#Wskazanie rodzaju grafu
def analyze_graph(G):
    print("Analiza grafu:")

    if isinstance(G, nx.DiGraph):
        print("Graf jest skierowany.")
    else:
        print("Graf jest nieskierowany.")

    if nx.is_weighted(G):
        print("Graf jest ważony.")
    else:
        print("Graf jest nieważony.")

    has_loops = any(u == v for u, v in G.edges)
    print(f"Graf {'zawiera pętle' if has_loops else 'nie zawiera pętli'}.")

    if isinstance(G, nx.DiGraph):
        is_connected = nx.is_weakly_connected(G)
    else:
        is_connected = nx.is_connected(G)
    print(f"Graf jest {'spójny' if is_connected else 'niespójny'}.")

analyze_graph(G_er)

#Wskazanie rzędu (liczba wierzchołków) oraz rozmiaru (liczba krawędzi)
print("Rząd grafu (liczba wierzchołków):", G_er.number_of_nodes())
print("Rozmiar grafu (liczba krawędzi):", G_er.number_of_edges())

#Wyliczenie miar centralności dla wszystkich wierzchołków tj. stopień (degree), bliskość (closeness), pośrednictwo (betweenness)
def centralities(G):
    print("Obliczanie miar centralności wierzchołków...")

    degree = nx.degree_centrality(G)
    closeness = nx.closeness_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight='weight')

    #Top 5 wierzchołków dla każdej miary
    print("\nTop 5 wierzchołków wg degree:")
    for node, value in sorted(degree.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"Wierzchołek {node}: {value:.4f}")

    print("\nTop 5 wierzchołków wg closeness:")
    for node, value in sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"Wierzchołek {node}: {value:.4f}")

    print("\nTop 5 wierzchołków wg betweenness:")
    for node, value in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"Wierzchołek {node}: {value:.4f}")

    return degree, closeness, betweenness

d, c, b = centralities(G_er)

#Wyliczenie miar centralności dla wszystkich krawędzi tj. pośrednictwo (betweenness)
def edge_betweenness(G):
    print("Obliczanie centralności pośrednictwa dla krawędzi...")

    edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight')

    #Top 5 najważniejszych krawędzi
    print("\nTop 5 krawędzi wg betweenness:")
    for edge, value in sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"Krawędź {edge}: {value:.4f}")

    return edge_betweenness

edge_b = edge_betweenness(G_er)

#wskazanie najważniejszych wierzchołków oraz krawędzi według wyliczonych miar centralności;

#<-

#Przedstawienie grafu w postaci macierzy incydencji i macierzy sąsiedztwa
def show_adjacency_and_incidence_matrices(G):
    print("Generowanie macierzy sąsiedztwa i incydencji...")

    #Macierz sąsiedztwa
    adj_matrix = nx.to_pandas_adjacency(G, dtype=int)
    print("\nMacierz sąsiedztwa (fragment):")
    print(adj_matrix.head())

    #Wizualizacja macierzy sąsiedztwa
    plt.figure(figsize=(10, 7))
    plt.imshow(adj_matrix, cmap='Blues', interpolation='nearest')
    plt.title("Macierz sąsiedztwa")
    plt.colorbar()
    plt.xlabel("Wierzchołki")
    plt.ylabel("Wierzchołki")
    plt.tight_layout()
    plt.show()

    #Macierz incydencji
    inc_matrix = nx.incidence_matrix(G, oriented=False).todense()
    inc_df = pd.DataFrame(inc_matrix, index=G.nodes(), columns=[f"e{ix}" for ix in range(G.number_of_edges())])
    print("\nMacierz incydencji (fragment):")
    print(inc_df.head())

    #Wizualizacja macierzy incydencji
    plt.figure(figsize=(10, 7))
    plt.imshow(inc_matrix, cmap='Blues', interpolation='nearest')
    plt.title("Macierz incydencji")
    plt.colorbar()
    plt.xlabel("Krawędzie")
    plt.ylabel("Wierzchołki")
    plt.tight_layout()
    plt.show()

    return adj_matrix, inc_df

adj, inc = show_adjacency_and_incidence_matrices(G_er)