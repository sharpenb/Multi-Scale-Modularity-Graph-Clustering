import os
import numpy as np
import networkx as nx


def load_dataset(directory, dataset_name):
    try:
        file = open(directory + dataset_name + "/type.txt", "r")
        graph_type = file.readline()[0:-1]
        file.close()
        if graph_type == "DW":
            G = nx.read_weighted_edgelist(directory + dataset_name + "/edge.txt", nodetype=int, create_using=nx.DiGraph())
        elif graph_type == "UW":
            G = nx.read_weighted_edgelist(directory + dataset_name + "/edge.txt", nodetype=int)
        elif graph_type == "DU":
            G = nx.read_edgelist(directory + dataset_name + "/edge.txt", nodetype=int, create_using=nx.DiGraph())
        else:
            G = nx.read_edgelist(directory + dataset_name + "/edge.txt", nodetype=int)
        G.name = dataset_name
        G = G.to_undirected()
    except:
        print("\nCannot load graph")
        G = nx.Graph()

    pos = {}
    try:
        file = open(directory + dataset_name + "/position.txt", "r")
        u = 0
        for line in file:
            s = line.split()
            pos[u] = (float(s[0]), float(s[1]))
            u += 1
        file.close()
        G.add_nodes_from(pos.keys())
    except:
        print("\nCannot load positions")

    label = {}
    try:
        file = open(directory + dataset_name + "/label.txt", "r")
        u = 0
        for line in file:
            label[u] = line[0:-1]
            u += 1
        file.close()
    except:
        print("\nCannot load labels")

    return G, pos, label


def save_dataset(directory, dataset_name, G, pos={}, label={}):

    if not os.path.exists(directory + dataset_name):
        os.mkdir(directory + dataset_name)

    file = open(directory + dataset_name + "/type.txt", "w")
    file.write("UW\n")
    file.close()

    file = open(directory + dataset_name + "/edge.txt", "w")
    for u, v in G.edges():
        file.write(str(u) + " " + str(v) + " " + str(G[u][v]['weight']) + "\n")
    file.close()

    if pos != {}:
        file = open(directory + dataset_name + "/position.txt", "w")
        for u in G.nodes():
            file.write(str(pos[u][0]) + " " + str(pos[u][1]) + " " + "\n")
    file.close()

    if label != {}:
        file = open(directory + dataset_name + "/label.txt", "w")
        for u in G.nodes():
            file.write(str(label[u]) + " " + "\n")
    file.close()


def clean_dataset(G, pos={}, label={}):
    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    G_cleaned = nx.relabel_nodes(G, mapping)
    pos_cleaned = {}
    for u in pos:
        pos_cleaned[mapping[u]] = pos[u]
    label_cleaned = {}
    for u in label:
        label_cleaned[mapping[u]] = label[u]

    return G_cleaned, pos_cleaned, label_cleaned


def extract_dataset_connected_components(G, pos={}, label={}, k=0):
    G_cc_k = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[k]
    pos_cc_k = {}
    if pos != {}:
        for u in G_cc_k.nodes():
            pos_cc_k[u] = pos[u]
    label_cc_k = {}
    if label != {}:
        for u in G_cc_k.nodes():
            label_cc_k[u] = label[u]

    return G_cc_k, pos_cc_k, label_cc_k


def connect_dataset_connected_components(G, pos={}, label={}):
    connected_components = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)
    giant_component = connected_components[0]
    min_weight = min(nx.get_edge_attributes(G, 'weight').values())
    new_edges = []
    for cc in connected_components:
        p = np.random.choice(giant_component.nodes())
        q = np.random.choice(cc.nodes())
        new_edges.append((p, q, min_weight))
    G.add_weighted_edges_from(new_edges)
    return G, pos, label


def generate_dataset_from_euclidean_points(points, similarity_measure, threshold, labels={}):
    G = nx.Graph()
    pos = {}
    edges = []
    for u, p in enumerate(points):
        G.add_node(u)
        pos[u] = p
        for v, q in enumerate(points):
            weight = similarity_measure(p, q)
            if weight > threshold:
                edges.append((u, v, weight))
    G.add_weighted_edges_from(edges)
    return G, pos, labels


