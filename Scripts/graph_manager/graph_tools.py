import numpy as np
import networkx as nx

def clusters_dict2clusters_list(cluster_dict):
    i = 0
    cluster_index = {}
    cluster_list = []
    for u, c in cluster_dict.items():
        if c not in cluster_index:
            cluster_list.append([u])
            cluster_index[c] = i
            i += 1
        else:
            cluster_list[cluster_index[c]].append(u)
    return cluster_list


def clusters_list2clusters_dict(cluster_list):
    cluster_dict = {}
    for i, c in enumerate(cluster_list):
        for u in c:
            cluster_dict[u] = i
    return cluster_dict


def modularity(G, clusters, resolution=1.):
    clusters_dict = clusters_list2clusters_dict(clusters)
    n_nodes = G.number_of_nodes()
    wtot = 0.
    w = {u: 0. for u in range(n_nodes)}
    for (u, v) in G.edges():
        weight = G[u][v]['weight']
        w[u] += weight
        w[v] += weight
        wtot += 2 * weight
    W = 0.
    S = 0.
    for u in G.nodes():
        for v in G.nodes():
            if clusters_dict[u] == clusters_dict[v]:
                S += w[u] * w[v]
                if G.has_edge(u,v):
                    W += G[u][v]['weight']
    W /= float(wtot)
    S /= float(wtot)**2
    return W - resolution * S
