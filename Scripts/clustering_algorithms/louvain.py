import networkx as nx
from graph_manager.graph_tools import clusters_dict2clusters_list
from graph_manager.plot_tools import *


def louvain(G, resolution=1, eps=0.001):
    clusters_dict = maximize(G, resolution, eps)
    n = len(clusters_dict)
    k = len(set(clusters_dict.values()))
    while k < n:
        H = aggregate(G, clusters_dict)
        new_cluster = maximize(H, resolution, eps)
        clusters_dict = {u: new_cluster[clusters_dict[u]] for u in G.nodes()}
        n = k
        k = len(set(clusters_dict.values()))
    return clusters_dict2clusters_list(clusters_dict)


def maximize(G, resolution, eps):
    # node weights
    node_weight = {u: 0. for u in G.nodes()}
    for (u, v) in G.edges():
        node_weight[u] += G[u][v]['weight']
        node_weight[v] += G[u][v]['weight']
    # total weight
    wtot = sum(list(node_weight.values()))
    # clusters
    cluster = {u: u for u in G.nodes()}
    # total weight of each cluster
    cluster_weight = {u: node_weight[u] for u in G.nodes()}
    # weights in each community to which the nodes are linked
    w = {u: {v: G[u][v]['weight'] for v in G.neighbors(u) if v != u} for u in G.nodes()}
    increase = True
    while increase:
        increase = False
        for u in G.nodes():
            # Compute delta for every neighbor
            delta = {}
            for k in w[u].keys():
                delta[k] = w[u][k] - resolution * node_weight[u] * cluster_weight[k] / wtot
            # Compute delta for u itself (if not already done)
            k = cluster[u]
            if k not in w[u].keys():
                delta[k] = - resolution * node_weight[u] * cluster_weight[k] / wtot
            # Compare the greatest delta to epsilon
            l = max(delta, key=delta.get)
            print u, k, l
            if delta[l] - delta[k] > resolution * (node_weight[u] * node_weight[u] / wtot) + eps / wtot:
                increase = True
                cluster[u] = l
                # Update information about neighbors and the community change of u
                cluster_weight[k] -= node_weight[u]
                cluster_weight[l] += node_weight[u]
                for v in G.neighbors(u):
                    if v != u:
                        w[v][k] -= G[u][v]['weight']
                        if w[v][k] == 0:
                            w[v].pop(k)
                        if l not in w[v].keys():
                            w[v][l] = 0
                        w[v][l] += G[u][v]['weight']
    return cluster


def aggregate(G, clusters_dict):
    H = nx.Graph()
    H.add_nodes_from(list(clusters_dict.values()))
    for (u,v) in G.edges():
        if H.has_edge(clusters_dict[u], clusters_dict[v]):
            H[clusters_dict[u]][clusters_dict[v]]['weight'] += G[u][v]['weight']
        else:
            H.add_edge(clusters_dict[u], clusters_dict[v])
            H[clusters_dict[u]][clusters_dict[v]]['weight'] = G[u][v]['weight']
    return H
