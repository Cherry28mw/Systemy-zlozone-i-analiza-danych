import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread
import random

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
'''
#Rozkład stopni wierzchołków
def degree_distribution(G):
    degrees = [d for n, d in G.degree()]
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1), alpha=0.75, color='blue')
    plt.title("Rozkład stopni wierzchołków")
    plt.xlabel("Stopień wierzchołka")
    plt.ylabel("Liczba wierzchołków")
    plt.show()

degree_distribution(G_ba)

#Gęstość sieci
density = nx.density(G_ba)
print(f"Gęstość sieci: {density:.4f}")

#Średnica sieci (najdłuższa ścieżka w grafie)
if nx.is_connected(G_ba):
    diameter = nx.diameter(G_ba)
    print(f"Średnica sieci: {diameter}")
else:
    print("Graf nie jest spójny, nie można obliczyć średnicy.")

#Średnia długość ścieżki
if nx.is_connected(G_ba):
    avg_path_length = nx.average_shortest_path_length(G_ba)
    print(f"Średnia długość ścieżki: {avg_path_length:.4f}")
else:
    print("Graf nie jest spójny, nie można obliczyć średniej długości ścieżki.")

#Najkrótsza ścieżka pomiędzy wskazanymi wierzchołkami
def shortest_path_between_nodes(G, node1, node2):
    try:
        path = nx.shortest_path(G, source=node1, target=node2)
        print(f"Najkrótsza ścieżka pomiędzy {node1} i {node2}: {path}")
    except nx.NetworkXNoPath:
        print(f"Brak ścieżki pomiędzy {node1} i {node2}")

#Przykład:
nodes = list(G_ba.nodes)
node1, node2 = random.sample(nodes, 2)
shortest_path_between_nodes(G_ba, node1, node2)

#Rozkład długości ścieżek
def path_length_distribution(G, max_nodes=1000):  # Dodajemy parametr max_nodes
    lengths = []

    # Jeśli graf jest niespójny, obliczamy dla każdej składowej spójnej
    components = list(nx.connected_components(G))

    # Przetwarzamy tylko pierwszą składową, aby ograniczyć obliczenia
    for component in components[:1]:  # Możesz zwiększyć, jeśli chcesz więcej składowych
        subgraph = G.subgraph(component)

        # Ograniczamy liczbę obliczeń do max_nodes (np. 1000 wierzchołków)
        nodes_to_process = list(subgraph.nodes)[:max_nodes]

        for source in nodes_to_process:
            for target in nodes_to_process:
                if source != target:
                    try:
                        path_length = nx.shortest_path_length(subgraph, source=source, target=target)
                        lengths.append(path_length)
                    except nx.NetworkXNoPath:
                        continue

    # Rysowanie histogramu, jeśli są dostępne długości ścieżek
    if lengths:
        plt.hist(lengths, bins=range(min(lengths), max(lengths) + 1), alpha=0.75, color='green')
        plt.title("Rozkład długości ścieżek")
        plt.xlabel("Długość ścieżki")
        plt.ylabel("Liczba ścieżek")
        plt.savefig("rozklad_dlugosci_sciezek.png", dpi=300)
        plt.show()

        print("Wykres zapisany jako rozklad_dlugosci_sciezek.png")
    else:
        print("Brak dostępnych ścieżek w grafie.")


# Przykład użycia funkcji z ograniczeniem liczby wierzchołków (1000)
path_length_distribution(G_ba, max_nodes=1000)


def path_length_distribution(G, max_nodes=1000):
    lengths = []

    #Sprawdzenie składowych spójnych grafu
    components = list(nx.connected_components(G))

    #Dla każdej składowej spójnej obliczenie najkrótszych ścieżek
    for component in components:
        subgraph = G.subgraph(component)
        for source in subgraph.nodes:
            for target in subgraph.nodes:
                if source != target:
                    try:
                        path_length = nx.shortest_path_length(subgraph, source=source, target=target)
                        lengths.append(path_length)
                    except nx.NetworkXNoPath:
                        continue

    #Histogram
    if lengths:
        plt.hist(lengths, bins=range(min(lengths), max(lengths) + 1), alpha=0.75, color='green')
        plt.title("Rozkład długości ścieżek")
        plt.xlabel("Długość ścieżki")
        plt.ylabel("Liczba ścieżek")
        plt.savefig("rozklad_dlugosci_sciezek.png", dpi=300)
        plt.show()

        print("Wykres zapisany jako rozklad_dlugosci_sciezek.png")
    else:
        print("Brak dostępnych ścieżek w grafie.")

#path_length_distribution(G_er)

#Rozkłady dla miar centralności wierzchołków i krawędzi
def centrality_distributions(G):
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    edge_betweenness = nx.edge_betweenness_centrality(G)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.hist(list(degree_centrality.values()), bins=30, alpha=0.75, color='blue')
    plt.title("Rozkład stopnia centralności")

    plt.subplot(1, 4, 2)
    plt.hist(list(closeness_centrality.values()), bins=30, alpha=0.75, color='red')
    plt.title("Rozkład bliskości centralności")

    plt.subplot(1, 4, 3)
    plt.hist(list(betweenness_centrality.values()), bins=30, alpha=0.75, color='green')
    plt.title("Rozkład pośrednictwa centralności")

    plt.subplot(1, 4, 4)
    plt.hist(list(edge_betweenness.values()), bins=30, alpha=0.75, color='purple')
    plt.title("Rozkład pośrednictwa krawędzi")
    plt.tight_layout()
    plt.savefig("miary_centralnosci_w_i_k.png", dpi=300)
    plt.show()

    print("Wykres zapisany jako miary_centralnosci_w_i_k.png")

#centrality_distributions(G_sub)

#PageRank
pagerank = nx.pagerank(G_ba)
node_sizes = [pagerank[node] * 10000 for node in G_ba.nodes()]  #Skalowanie rozmiaru wierzchołków

#Przygotowanie kolorów wierzchołków na podstawie PageRank
node_colors = [pagerank[node] for node in G_ba.nodes()]

#Tworzenie układu wierzchołków
print("Obliczanie układu wierzchołków...")
pos = nx.spring_layout(G_ba, seed=42)

#Rysowanie grafu z PageRank jako rozmiar i kolor wierzchołków
plt.figure(figsize=(14, 14))
nx.draw_networkx_nodes(G_ba, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.Blues, alpha=0.7)
nx.draw_networkx_edges(G_ba, pos, width=0.5, alpha=0.3)
nx.draw_networkx_labels(G_ba, pos, font_size=8)
plt.title("Wizualizacja grafu z PageRank", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.savefig("PageRank.png", dpi=300)
plt.show()

print("Wykres zapisany jako PageRank.png")

#Składowe spójne
connected_components = list(nx.connected_components(G_ba))

#Ustawienie kolorów dla różnych składowych
colors = plt.cm.rainbow(np.linspace(0, 1, len(connected_components)))

#Tworzenie układu wierzchołków
pos = nx.spring_layout(G_ba, seed=42)

#Graf
plt.figure(figsize=(14, 14))
for i, component in enumerate(connected_components):
    component_subgraph = G_ba.subgraph(component)
    nx.draw_networkx_nodes(component_subgraph, pos, node_size=50, node_color=[colors[i]], label=f"Składowa {i + 1}")
    nx.draw_networkx_edges(component_subgraph, pos, alpha=0.5)
nx.draw_networkx_labels(G_ba, pos, font_size=8, font_color="black")
plt.title("Wizualizacja składowych spójnych w grafie", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.savefig("skladowe_spojne.png", dpi=300)
plt.show()

print("Wykres zapisany jako skladowe_spojne.png")

#K-spójność
def k_connectivity(G):
    k_conn = nx.edge_connectivity(G)
    print(f"Graf ma k-spójność krawędziową równą: {k_conn}")

k_connectivity(G_ba)

#Kategorie węzłów (huby, mosty, przeguby)
def node_categories(G):
    #Huby (wierzchołki o wysokim stopniu centralności)
    hubs = [node for node, degree in G.degree() if degree > 5]

    #Mosty (krawędzie, których usunięcie podzieliłoby graf na dwie składowe)
    bridges = list(nx.bridges(G))

    #Przeguby (wierzchołki o wysokiej centralności pośrednictwa)
    betweenness_centrality = nx.betweenness_centrality(G)
    high_betweenness = [node for node, bc in betweenness_centrality.items() if bc > 0.01]  # Możesz dostosować próg

    #Tworzenie układu wierzchołków
    pos = nx.spring_layout(G, seed=42)

    #Graf
    plt.figure(figsize=(14, 14))
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='gray', alpha=0.5)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3)
    #Rysowanie hubów (większe wierzchołki)
    nx.draw_networkx_nodes(G, pos, nodelist=hubs, node_size=200, node_color='red', label="Huby")
    #Rysowanie mostów (grubsze krawędzie)
    nx.draw_networkx_edges(G, pos, edgelist=bridges, width=2, alpha=0.7, edge_color='blue', label="Mosty")
    #Rysowanie przegubów (inne kolory wierzchołków)
    nx.draw_networkx_nodes(G, pos, nodelist=high_betweenness, node_size=100, node_color='green', label="Przeguby")
    plt.legend()
    plt.title("Wizualizacja kategorii węzłów: Huby, Mosty, Przeguby", fontsize=16)
    plt.axis('off')  # Ukrycie osi
    plt.tight_layout()
    plt.savefig("huby_mosty_przegoby.png", dpi=300)
    plt.show()

    print("Wykres zapisany jako huby_mosty_przegoby.png")

node_categories(G_ba)
'''
# Kliki
def cliques(G):
    #Kliki n-tego rzędu (np. 3-kliki)
    cliques = list(nx.find_cliques(G))

    #Maksymalna klika
    max_clique = max(cliques, key=len)
    print(f"Maksymalna klika: {max_clique}")

    #Near cliques (wierzchołki w pobliżu kliki)
    near_cliques = [clique for clique in cliques if len(clique) > 3]
    print(f"Near cliques: {near_cliques}")

    #Tworzenie układu wierzchołków
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(14, 14))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Rysowanie podstawowego grafu
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color='lightgray', alpha=0.3)
    nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.2)

    # Rysowanie maksymalnej kliki
    nx.draw_networkx_nodes(G, pos, nodelist=max_clique, node_size=200, node_color='red', label="Maksymalna klika")

    # Rysowanie kilku near-cliques (ograniczenie)
    max_near_cliques = near_cliques[:10]  # tylko 10 pierwszych, możesz zmienić
    for idx, clique in enumerate(max_near_cliques):
        nx.draw_networkx_nodes(G, pos, nodelist=clique, node_size=80, node_color='green', alpha=0.5)

    plt.title("Kliki w grafie: maksymalna + kilka near-cliques", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("kliki.png", dpi=200)
    plt.show()

    print("Wykres zapisany jako kliki.png")


'''
    # Rysowanie grafu
    plt.figure(figsize=(18, 18))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # Rysowanie wszystkich wierzchołków
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='gray', alpha=0.5)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3)
    # Rysowanie klik
    for clique in cliques:
        nx.draw_networkx_nodes(G, pos, nodelist=clique, node_size=200, node_color='blue', alpha=0.7)
    # Rysowanie maksymalnej klik (większe wierzchołki)
    nx.draw_networkx_nodes(G, pos, nodelist=max_clique, node_size=500, node_color='red', label="Maksymalna klika")
    # Rysowanie near cliques
    for clique in near_cliques:
        nx.draw_networkx_nodes(G, pos, nodelist=clique, node_size=100, node_color='green', alpha=0.6, label="Near clique")

    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")
    plt.title("Wizualizacja klik w grafie: Kliki, Maksymalna klika, Near cliques", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("kliki.png", dpi=300)
    plt.show()

    print("Wykres zapisany jako kliki.png")
'''
cliques(G_er)
