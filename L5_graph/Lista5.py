import os
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score

data_path = "WRUT_Okna"
random.seed(42)
np.random.seed(42)

# Wczytanie plików
o1p1_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.startswith("o1p1_")])
o7p7_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.startswith("o7p7_")])

# Funkcje pomocnicze
def count_edges(file_list):
    return [sum(1 for _ in open(f)) for f in file_list]

def load_graph(filepaths):
    G = nx.Graph()
    for filepath in filepaths:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) >= 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        G.add_edge(u, v)
                    except ValueError:
                        continue
    return G

def precision_recall(predicted, actual):
    predicted_set = set(predicted)
    actual_set = set((min(u,v), max(u,v)) for u, v in actual)
    tp = len(predicted_set & actual_set)
    precision = tp / len(predicted_set) if predicted_set else 0
    recall = tp / len(actual_set) if actual_set else 0
    return precision, recall

def generate_negative_samples(G, pos_edges, n):
    nodes = list(G.nodes())
    neg_edges = set()
    while len(neg_edges) < n:
        u, v = random.sample(nodes, 2)
        if u != v and not G.has_edge(u, v) and (u, v) not in pos_edges and (v, u) not in pos_edges:
            neg_edges.add((u, v))
    return list(neg_edges)

def plot_edge_trends(o1p1_counts, o7p7_counts):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(o1p1_counts)), o1p1_counts, label="o1p1 (okna 1-dniowe)")
    plt.plot(range(len(o7p7_counts)), o7p7_counts, label="o7p7 (okna 7-dniowe)")
    plt.xlabel("Numer okna czasowego")
    plt.ylabel("Liczba połączeń (krawędzi)")
    plt.title("Liczba połączeń w czasie")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_predictions(G_train, predicted_edges, title, centrality=False):
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G_train, seed=42)

    # Węzły: kolor wg degree lub niebieski
    if centrality:
        degrees = dict(G_train.degree())
        node_color = [degrees[n] for n in G_train.nodes()]
        nx.draw_networkx_nodes(G_train, pos, node_size=30, cmap=plt.cm.viridis, node_color=node_color)
    else:
        nx.draw_networkx_nodes(G_train, pos, node_size=30, node_color='lightblue')

    # Krawędzie istniejące
    nx.draw_networkx_edges(G_train, pos, alpha=0.2)

    # Przewidywane nowe krawędzie
    nx.draw_networkx_edges(G_train, pos, edgelist=predicted_edges, edge_color='red', width=1.5, alpha=0.7)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Główna predykcja i wizualizacja
def predict_and_evaluate(train_files, test_file, label):
    G_train = load_graph(train_files)
    G_test = load_graph([test_file])

    test_edges = [
        (u, v) for u, v in G_test.edges()
        if not G_train.has_edge(u, v) and G_train.has_node(u) and G_train.has_node(v)
    ]
    true_count = len(test_edges)

    jaccard = sorted(nx.jaccard_coefficient(G_train, test_edges), key=lambda x: x[2], reverse=True)
    adamic = sorted(nx.adamic_adar_index(G_train, test_edges), key=lambda x: x[2], reverse=True)

    pred_jc = [(min(u, v), max(u, v)) for u, v, _ in jaccard[:true_count]]
    pred_aa = [(min(u, v), max(u, v)) for u, v, _ in adamic[:true_count]]

    p_jc, r_jc = precision_recall(pred_jc, test_edges)
    p_aa, r_aa = precision_recall(pred_aa, test_edges)

    print(f"\n[{label}] Klasyczne podejście (top-{true_count})")
    print(f"Jaccard - Precision: {p_jc:.4f}, Recall: {r_jc:.4f}")
    print(f"Adamic - Precision: {p_aa:.4f}, Recall: {r_aa:.4f}")

    # Wizualizacja
    visualize_predictions(G_train, pred_jc, f"[{label}] Przewidywane połączenia - Jaccard", centrality=True)
    visualize_predictions(G_train, pred_aa, f"[{label}] Przewidywane połączenia - Adamic-Adar", centrality=True)

    # Klasyfikacja binarna
    neg_edges = generate_negative_samples(G_train, test_edges, len(test_edges))
    all_edges = test_edges + neg_edges
    y_true = [1]*len(test_edges) + [0]*len(neg_edges)

    j_all = list(nx.jaccard_coefficient(G_train, all_edges))
    a_all = list(nx.adamic_adar_index(G_train, all_edges))

    j_scores = [s for _, _, s in j_all]
    a_scores = [s for _, _, s in a_all]

    y_pred_j = [1 if s > 0 else 0 for s in j_scores]
    y_pred_a = [1 if s > 0 else 0 for s in a_scores]

    auc_j = roc_auc_score(y_true, j_scores)
    auc_a = roc_auc_score(y_true, a_scores)

    prec_j = precision_score(y_true, y_pred_j)
    prec_a = precision_score(y_true, y_pred_a)
    rec_j = recall_score(y_true, y_pred_j)
    rec_a = recall_score(y_true, y_pred_a)

    print(f"\n[{label}] Klasyfikacja binarna")
    print(f"Jaccard - AUC: {auc_j:.4f}, Precision: {prec_j:.4f}, Recall: {rec_j:.4f}")
    print(f"Adamic - AUC: {auc_a:.4f}, Precision: {prec_a:.4f}, Recall: {rec_a:.4f}")

def predict_and_evaluate_realistic(train_files, test_file, label):
    G_train = load_graph(train_files)
    G_test = load_graph([test_file])

    true_new_edges = [
        (min(u, v), max(u, v)) for u, v in G_test.edges()
        if not G_train.has_edge(u, v)
    ]

    possible_pairs = list(nx.non_edges(G_train))
    possible_pairs = [
        (u, v) for u, v in possible_pairs
        if G_train.has_node(u) and G_train.has_node(v)
    ]

    jaccard_all = sorted(nx.jaccard_coefficient(G_train, possible_pairs), key=lambda x: x[2], reverse=True)
    adamic_all = sorted(nx.adamic_adar_index(G_train, possible_pairs), key=lambda x: x[2], reverse=True)

    k = len(true_new_edges)
    pred_jc = [(min(u, v), max(u, v)) for u, v, _ in jaccard_all[:k]]
    pred_aa = [(min(u, v), max(u, v)) for u, v, _ in adamic_all[:k]]

    prec_jc, rec_jc = precision_recall(pred_jc, true_new_edges)
    prec_aa, rec_aa = precision_recall(pred_aa, true_new_edges)

    print(f"\n[{label}] Realistyczna predykcja (top-{k})")
    print(f"Jaccard - Precision: {prec_jc:.4f}, Recall: {rec_jc:.4f}")
    print(f"Adamic - Precision: {prec_aa:.4f}, Recall: {rec_aa:.4f}")

# --- Wykres zmian liczby krawędzi w czasie ---
o1p1_counts = count_edges(o1p1_files)
o7p7_counts = count_edges(o7p7_files)
plot_edge_trends(o1p1_counts, o7p7_counts)

# Predykcje i wizualizacje ---
predict_and_evaluate(o1p1_files[:100], o1p1_files[101], label="o1p1")
predict_and_evaluate(o7p7_files[:10], o7p7_files[11], label="o7p7")

predict_and_evaluate_realistic(o1p1_files[:100], o1p1_files[101], label="o1p1")
predict_and_evaluate_realistic(o7p7_files[:10], o7p7_files[11], label="o7p7")
