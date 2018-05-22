from graph_manager.graph_tools import clusters_list2clusters_dict


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