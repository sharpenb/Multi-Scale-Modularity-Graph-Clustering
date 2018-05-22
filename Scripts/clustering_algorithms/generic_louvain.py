import networkx as nx
import numpy as np
from graph_manager.graph_tools import clusters_dict2clusters_list
from graph_manager.plot_tools import *

def generic_louvain(G, resolution=1, eps=0.001, modular_function=lambda pi, pj, pij, resolution: pij - resolution * pi *pj):
    clusters_dict = maximize(G, resolution, eps)
    n = len(clusters_dict)
    k = len(set(clusters_dict.values()))
    # while k < n:
    #     H = aggregate(G, clusters_dict)
    #     new_cluster = maximize(H, resolution, eps, modular_function)
    #     clusters_dict = {u: new_cluster[clusters_dict[u]] for u in G.nodes()}
    #     n = k
    #     k = len(set(clusters_dict.values()))
    return clusters_dict2clusters_list(clusters_dict)


def maximize(G, resolution, eps, modular_function=lambda pij, pi, pj, resolution: pij - resolution * pi * pj):
    node_weight = {u: 0. for u in G.nodes()}
    for (u, v) in G.edges():
        node_weight[u] += G[u][v]['weight']
        node_weight[v] += G[u][v]['weight']
    wtot = sum(list(node_weight.values()))
    cluster = {u: u for u in G.nodes()}
    cluster_comp = {u: set([u]) for u in G.nodes()}
    W = {u: {v: G[u][v]['weight'] for v in G.neighbors(u)} for u in G.nodes()}
    S = {}
    increase = True
    while increase:
        increase = False
        for u in G.nodes():
            delta = {}

            S[u] = {}
            for v in G.neighbors(u):
                if cluster[v] not in S[u]:
                    S[u][cluster[v]] = 0
                    for w in cluster_comp[cluster[v]]:
                        if w != u:
                            if w in G.neighbors(u):
                                S[u][cluster[v]] += modular_function(G[u][w]['weight'] / wtot, node_weight[u] / wtot, node_weight[w] / wtot, resolution)
                            else:
                                S[u][cluster[v]] += 0 #modular_function(1., node_weight[u] / wtot, node_weight[w] / wtot, resolution)


            for k in S[u]:
                # delta[k] = W[u][k]/float(wtot) - resolution * S[u][k]
                delta[k] = S[u][k]
            k = cluster[u]
            if k not in S[u]:
                delta[k] = - resolution * 0.
            l = max(delta, key=delta.get)

            if delta[l] - delta[k] > eps / wtot:
                increase = True
                cluster[u] = l
                cluster_comp[k].remove(u)
                cluster_comp[l].add(u)

                # for v in G.neighbors(u):
                #     if v != u:
                #         W[v][k] -= G[u][v]['weight']
                #         if W[v][k] == 0:
                #             W[v].pop(k)
                #         if l not in W[v]:
                #             W[v][l] = 0
                #         W[v][l] += G[u][v]['weight']

    return cluster


# def maximize(G, resolution, eps):
#     node_weight = {u: 0. for u in G.nodes()}
#     for (u, v) in G.edges():
#         node_weight[u] += G[u][v]['weight']
#         node_weight[v] += G[u][v]['weight']
#     wtot = sum(list(node_weight.values()))
#     cluster = {u: u for u in G.nodes()}
#     cluster_comp = {u: set([u]) for u in G.nodes()}
#     cluster_neighbors = {u: set([v for v in G.neighbors(u)]) for u in G.nodes()}
#     cluster_weight = {u: node_weight[u] for u in G.nodes()}
#     W = {u: {v: G[u][v]['weight'] for v in G.neighbors(u)} for u in G.nodes()}
#     S = {u: {v: (node_weight[u] * node_weight[v]) / float(wtot*wtot) for v in G.neighbors(u)} for u in G.nodes()}
#     increase = True
#     while increase:
#         increase = False
#         for u in G.nodes():
#             delta = {}
#             for k in W[u]:
#                 delta[k] = W[u][k]/float(wtot) - resolution * S[u][k]
#             k = cluster[u]
#             if k not in W[u]:
#                 delta[k] = - resolution * 0.
#             l = max(delta, key=delta.get)
#             print u, k, l
#             if delta[l] - delta[k] > eps / wtot:
#                 increase = True
#                 cluster[u] = l
#                 cluster_weight[k] -= node_weight[u]
#                 cluster_weight[l] += node_weight[u]
#                 cluster_comp[k].remove(u)
#
#                 for v in G.neighbors(u):
#                     if v != u:
#                         W[v][k] -= G[u][v]['weight']
#                         if W[v][k] == 0:
#                             W[v].pop(k)
#                         if l not in W[v]:
#                             W[v][l] = 0
#                         W[v][l] += G[u][v]['weight']
#
#                 disconnected = set([])
#                 for neighboring_cluster in cluster_neighbors[k]:
#                     for v in cluster_comp[neighboring_cluster]:
#                         S[v][k] -= (node_weight[u] * node_weight[v]) / float(wtot * wtot)
#                         if S[v][k] == 0:
#                             S[v].pop(k)
#                             if neighboring_cluster not in disconnected:
#                                 disconnected.add(neighboring_cluster)
#
#                 # TODO: Manage neighborhood of clusters
#                 for c in disconnected:
#                     if k != c:
#                         cluster_neighbors[c].remove(k)
#                         cluster_neighbors[k].remove(c)
#                 for v in G.neighbors(u):
#                     if l not in cluster_neighbors[cluster[v]]:
#                         cluster_neighbors[cluster[v]].add(l)
#                         cluster_neighbors[l].add(cluster[v])
#
#                 for neighboring_cluster in cluster_neighbors[l]:
#                     for v in cluster_comp[neighboring_cluster]:
#                         if l not in S[v]:
#                             S[v][l] = 0
#                         S[v][l] += (node_weight[u] * node_weight[v]) / float(wtot * wtot)
#
#                 cluster_comp[l].add(u)
#
#     return cluster


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
