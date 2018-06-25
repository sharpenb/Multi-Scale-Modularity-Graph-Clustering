import numpy as np
import networkx as nx

_AFFINITY = {'unitary', 'weighted'}
_LINKAGE = {'single', 'average', 'complete', 'modular', 'classic', 'classic-uniform',  'classic-weighted', 'KL-uniform', 'KL-weighted'}
_DIVERGENCE = {'Kullback-Leibler'}


def graph2tree_cost(graph, dendrogram, affinity='weighted', linkage='classic', g=lambda a, b: a + b, check=True):

    if linkage not in _LINKAGE:
        raise ValueError("Unknown linkage type %s."
                         "Valid options are %s" % (linkage, _LINKAGE.keys()))

    graph_copy = graph.copy()

    if check:

        graph_copy = nx.convert_node_labels_to_integers(graph_copy)

        if affinity == 'unitary':
            for e in graph_copy.edges():
                graph_copy.add_edge(e[0], e[1], weight=1)

        n_edges = len(graph_copy.edges())
        n_weighted_edges = len(nx.get_edge_attributes(graph_copy, 'weight'))
        if affinity == 'weighted' and not n_weighted_edges == n_edges:
            raise KeyError("%s edges among %s do not have the attribute/key \'weight\'."
                           % (n_edges - n_weighted_edges, n_edges))

        n_nodes = len(graph_copy.nodes())
        dendrogram_size = np.shape(dendrogram)[0]
        if n_nodes != dendrogram_size + 1:
            raise ValueError('The graph size (%s) and the dendrogram size (%s) do not match' % (n_nodes, dendrogram_size))

    if linkage == 'single':
        cost = single_linkage_hierarchical_cost(graph_copy, dendrogram, g)
    elif linkage == 'average':
        cost = average_linkage_hierarchical_cost(graph_copy, dendrogram, g)
    elif linkage == 'complete':
        cost = complete_linkage_hierarchical_cost(graph_copy, dendrogram, g)
    elif linkage == 'modular':
        cost = modular_linkage_hierarchical_cost(graph_copy, dendrogram, g)
    elif linkage == 'classic':
        cost = classic_linkage_hierarchical_cost(graph_copy, dendrogram, g)
    elif linkage == 'classic-uniform':
        cost = uniform_tree_cost(graph_copy, dendrogram, g)
    elif linkage == 'classic-weighted':
        cost = weighted_tree_cost(graph_copy, dendrogram, g)
    elif linkage == 'KL-uniform':
        cost = kullback_leibler_uniform_tree_cost(graph_copy, dendrogram, g)
    elif linkage == 'KL-weighted':
        cost = kullback_leibler_weighted_tree_cost(graph_copy, dendrogram, g)

    return cost


def single_linkage_hierarchical_cost(graph, dendrogram, g):
    n_nodes = np.shape(dendrogram)[0] + 1

    cost = 0

    w = {u: 0 for u in range(n_nodes)}
    wtot = 0
    for (u, v) in graph.edges():
        weight = graph[u][v]['weight']
        w[u] += weight
        w[v] += weight
        wtot += 2 * weight
    u = n_nodes

    cluster_size = {t: 1 for t in range(n_nodes)}
    for t in range(n_nodes - 1):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])

        linkage = graph[a][b]['weight']
        cost += linkage * g(cluster_size[a], cluster_size[b])

        graph.add_node(u)
        neighbors_a = list(graph.neighbors(a))
        neighbors_b = list(graph.neighbors(b))
        for v in neighbors_a:
            graph.add_edge(u, v, weight=graph[a][v]['weight'])
        for v in neighbors_b:
            if graph.has_edge(u, v):
                graph[u][v]['weight'] = max(graph[b][v]['weight'], graph[u][v]['weight'])
            else:
                graph.add_edge(u, v, weight=graph[b][v]['weight'])
        graph.remove_node(a)
        graph.remove_node(b)
        w[u] = w.pop(a) + w.pop(b)
        cluster_size[u] = cluster_size.pop(a) + cluster_size.pop(b)
        u += 1

    return cost


def average_linkage_hierarchical_cost(graph, dendrogram, g):
    n_nodes = np.shape(dendrogram)[0] + 1

    cost = 0

    w = {u: 0 for u in range(n_nodes)}
    wtot = 0
    for (u, v) in graph.edges():
        weight = graph[u][v]['weight']
        w[u] += weight
        w[v] += weight
        wtot += 2 * weight
    u = n_nodes

    cluster_size = {t: 1 for t in range(n_nodes)}
    for t in range(n_nodes - 1):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])

        linkage = graph[a][b]['weight'] / float(cluster_size[a] * cluster_size[b])
        cost += linkage * g(cluster_size[a], cluster_size[b])

        graph.add_node(u)
        neighbors_a = list(graph.neighbors(a))
        neighbors_b = list(graph.neighbors(b))
        for v in neighbors_a:
            graph.add_edge(u, v, weight=graph[a][v]['weight'])
        for v in neighbors_b:
            if graph.has_edge(u, v):
                graph[u][v]['weight'] += graph[b][v]['weight']
            else:
                graph.add_edge(u, v, weight=graph[b][v]['weight'])
        graph.remove_node(a)
        graph.remove_node(b)
        w[u] = w.pop(a) + w.pop(b)
        cluster_size[u] = cluster_size.pop(a) + cluster_size.pop(b)
        u += 1

    return cost


def complete_linkage_hierarchical_cost(graph, dendrogram, g):
    n_nodes = np.shape(dendrogram)[0] + 1

    cost = 0

    u = n_nodes

    cluster_size = {t: 1 for t in range(n_nodes)}
    for t in range(n_nodes - 1):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])

        linkage = graph[a][b]['weight']
        cost += linkage * g(cluster_size[a], cluster_size[b])

        graph.add_node(u)
        neighbors_a = list(graph.neighbors(a))
        neighbors_b = list(graph.neighbors(b))
        for v in neighbors_a:
            graph.add_edge(u, v, weight=graph[a][v]['weight'])
        for v in neighbors_b:
            if graph.has_edge(u, v):
                graph[u][v]['weight'] = min(graph[b][v]['weight'], graph[u][v]['weight'])
            else:
                graph.add_edge(u, v, weight=graph[b][v]['weight'])
        graph.remove_node(a)
        graph.remove_node(b)
        cluster_size[u] = cluster_size.pop(a) + cluster_size.pop(b)
        u += 1

    return cost


def modular_linkage_hierarchical_cost(graph, dendrogram, g):
    n_nodes = np.shape(dendrogram)[0] + 1

    cost = 0

    w = {u: 0 for u in range(n_nodes)}
    wtot = 0
    for (u, v) in graph.edges():
        weight = graph[u][v]['weight']
        w[u] += weight
        w[v] += weight
        wtot += 2 * weight
    u = n_nodes

    cluster_size = {t: 1 for t in range(n_nodes)}
    for t in range(n_nodes - 1):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])

        linkage = graph[a][b]['weight'] # / float(w[a] * w[b])
        cost += linkage * g(w[a], w[b]) #* g(cluster_size[a], cluster_size[b])

        graph.add_node(u)
        neighbors_a = list(graph.neighbors(a))
        neighbors_b = list(graph.neighbors(b))
        for v in neighbors_a:
            graph.add_edge(u, v, weight=graph[a][v]['weight'])
        for v in neighbors_b:
            if graph.has_edge(u, v):
                graph[u][v]['weight'] += graph[b][v]['weight']
            else:
                graph.add_edge(u, v, weight=graph[b][v]['weight'])
        graph.remove_node(a)
        graph.remove_node(b)
        w[u] = w.pop(a) + w.pop(b)
        cluster_size[u] = cluster_size.pop(a) + cluster_size.pop(b)
        u += 1

    return cost


def uniform_tree_cost(graph, dendrogram, g):
    n_nodes = np.shape(dendrogram)[0] + 1

    cost = 0

    w = {u: 0 for u in range(n_nodes)}
    wtot = 0
    for (u, v) in graph.edges():
        weight = graph[u][v]['weight']
        w[u] += weight
        w[v] += weight
        wtot += 2 * weight
    u = n_nodes

    cluster_size = {t: 1 for t in range(n_nodes)}
    for t in range(n_nodes - 1):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])

        if graph.has_edge(a, b):
            linkage = 2 * graph[a][b]['weight'] / float(wtot)
        else:
            linkage = 0
        cost += linkage * g(cluster_size[a] / float(n_nodes), cluster_size[b] / float(n_nodes))

        graph.add_node(u)
        neighbors_a = list(graph.neighbors(a))
        neighbors_b = list(graph.neighbors(b))
        for v in neighbors_a:
            graph.add_edge(u, v, weight=graph[a][v]['weight'])
        for v in neighbors_b:
            if graph.has_edge(u, v):
                graph[u][v]['weight'] += graph[b][v]['weight']
            else:
                graph.add_edge(u, v, weight=graph[b][v]['weight'])
        graph.remove_node(a)
        graph.remove_node(b)
        w[u] = w.pop(a) + w.pop(b)
        cluster_size[u] = cluster_size.pop(a) + cluster_size.pop(b)
        u += 1

    return cost


def weighted_tree_cost(graph, dendrogram, g):
    n_nodes = np.shape(dendrogram)[0] + 1

    cost = 0

    w = {u: 0 for u in range(n_nodes)}
    wtot = 0
    for (u, v) in graph.edges():
        weight = graph[u][v]['weight']
        w[u] += weight
        w[v] += weight
        wtot += 2 * weight
    u = n_nodes

    cluster_size = {t: 1 for t in range(n_nodes)}
    for t in range(n_nodes - 1):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])

        if graph.has_edge(a, b):
            linkage = 2 * graph[a][b]['weight'] / float(wtot)
        else:
            linkage = 0
        cost += linkage * g(w[a] / float(wtot), w[b] / float(wtot))

        graph.add_node(u)
        neighbors_a = list(graph.neighbors(a))
        neighbors_b = list(graph.neighbors(b))
        for v in neighbors_a:
            graph.add_edge(u, v, weight=graph[a][v]['weight'])
        for v in neighbors_b:
            if graph.has_edge(u, v):
                graph[u][v]['weight'] += graph[b][v]['weight']
            else:
                graph.add_edge(u, v, weight=graph[b][v]['weight'])
        graph.remove_node(a)
        graph.remove_node(b)
        w[u] = w.pop(a) + w.pop(b)
        cluster_size[u] = cluster_size.pop(a) + cluster_size.pop(b)
        u += 1

    return cost


def kullback_leibler_uniform_tree_cost(graph, dendrogram, g):
    n_nodes = np.shape(dendrogram)[0] + 1

    cost = 0

    w = {u: 0 for u in range(n_nodes)}
    wtot = 0
    for (u, v) in graph.edges():
        weight = graph[u][v]['weight']
        w[u] += weight
        w[v] += weight
        wtot += 2 * weight
    u = n_nodes

    cluster_size = {t: 1 for t in range(n_nodes)}
    for t in range(n_nodes - 1):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])

        if graph.has_edge(a, b):
            p_ab = 2 * graph[a][b]['weight'] / float(wtot)
        else:
            p_ab = 0
        p_a = cluster_size[a] / float(n_nodes)
        p_b = cluster_size[b] / float(n_nodes)
        cost += p_ab * np.log(p_a * p_b / p_ab)

        graph.add_node(u)
        neighbors_a = list(graph.neighbors(a))
        neighbors_b = list(graph.neighbors(b))
        for v in neighbors_a:
            graph.add_edge(u, v, weight=graph[a][v]['weight'])
        for v in neighbors_b:
            if graph.has_edge(u, v):
                graph[u][v]['weight'] += graph[b][v]['weight']
            else:
                graph.add_edge(u, v, weight=graph[b][v]['weight'])
        graph.remove_node(a)
        graph.remove_node(b)
        w[u] = w.pop(a) + w.pop(b)
        cluster_size[u] = cluster_size.pop(a) + cluster_size.pop(b)
        u += 1

    return cost


def kullback_leibler_weighted_tree_cost(graph, dendrogram, g):
    n_nodes = np.shape(dendrogram)[0] + 1

    cost = 0

    w = {u: 0 for u in range(n_nodes)}
    wtot = 0
    for (u, v) in graph.edges():
        weight = graph[u][v]['weight']
        w[u] += weight
        w[v] += weight
        wtot += 2 * weight
    u = n_nodes

    cluster_size = {t: 1 for t in range(n_nodes)}
    for t in range(n_nodes - 1):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])

        if graph.has_edge(a, b):
            p_ab = 2 * graph[a][b]['weight'] / float(wtot)
        else:
            p_ab = 0
        p_a = w[a] / float(wtot)
        p_b = w[b] / float(wtot)
        cost += p_ab * np.log(p_a * p_b / p_ab)

        graph.add_node(u)
        neighbors_a = list(graph.neighbors(a))
        neighbors_b = list(graph.neighbors(b))
        for v in neighbors_a:
            graph.add_edge(u, v, weight=graph[a][v]['weight'])
        for v in neighbors_b:
            if graph.has_edge(u, v):
                graph[u][v]['weight'] += graph[b][v]['weight']
            else:
                graph.add_edge(u, v, weight=graph[b][v]['weight'])
        graph.remove_node(a)
        graph.remove_node(b)
        w[u] = w.pop(a) + w.pop(b)
        cluster_size[u] = cluster_size.pop(a) + cluster_size.pop(b)
        u += 1

    return cost


def classic_linkage_hierarchical_cost(graph, dendrogram, g):
    n_nodes = np.shape(dendrogram)[0] + 1

    cost = 0

    u = n_nodes

    cluster_size = {t: 1 for t in range(n_nodes)}
    for t in range(n_nodes - 1):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])

        linkage = graph[a][b]['weight']
        cost += linkage * g(cluster_size[a], cluster_size[b])

        graph.add_node(u)
        neighbors_a = list(graph.neighbors(a))
        neighbors_b = list(graph.neighbors(b))
        for v in neighbors_a:
            graph.add_edge(u, v, weight=graph[a][v]['weight'])
        for v in neighbors_b:
            if graph.has_edge(u, v):
                graph[u][v]['weight'] += graph[b][v]['weight']
            else:
                graph.add_edge(u, v, weight=graph[b][v]['weight'])
        graph.remove_node(a)
        graph.remove_node(b)
        cluster_size[u] = cluster_size.pop(a) + cluster_size.pop(b)
        u += 1

    return cost


def graph2dendrogram_cost(graph, dendrogram, affinity="weighted", divergence="Kullback-Leibler", check=True):

    if divergence not in _DIVERGENCE:
        raise ValueError("Unknown divergence type %s."
                         "Valid options are %s" % (divergence, _DIVERGENCE))

    graph_copy = graph.copy()

    if check:
        graph_copy = nx.convert_node_labels_to_integers(graph_copy)

        if affinity == 'unitary':
            for e in graph_copy.edges:
                graph_copy.add_edge(e[0], e[1], weight=1)

        n_edges = len(graph_copy.edges())
        n_weighted_edges = len(nx.get_edge_attributes(graph_copy, 'weight'))
        if affinity == 'weighted' and not n_weighted_edges == n_edges:
            raise KeyError("%s edges among %s do not have the attribute/key \'weigth\'."
                           % (n_edges - n_weighted_edges, n_edges))

        n_nodes = len(graph_copy.nodes())
        dendrogram_size = np.shape(dendrogram)[0]
        if n_nodes != dendrogram_size + 1:
            raise ValueError('The graph size (%s) and the dendrogram size (%s) do not match' % (n_nodes, dendrogram_size))

    if divergence == 'Kullback-Leibler':
        cost = kullback_leibler_dendrogram_cost(graph_copy, dendrogram)

    return cost


def kullback_leibler_dendrogram_cost(graph, dendrogram):
    n_nodes = np.shape(dendrogram)[0] + 1

    objective_sum__term = 0
    objective_log_term = 0

    w = {u: 0 for u in range(n_nodes)}
    wtot = 0
    for (u, v) in graph.edges():
        weight = graph[u][v]['weight']
        w[u] += weight
        w[v] += weight
        wtot += 2 * weight
    u = n_nodes

    for t in range(n_nodes - 1):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])

        p_ab = 2 * graph[a][b]['weight'] / float(wtot)
        p_a = w[a] / float(wtot)
        p_b = w[b] / float(wtot)
        d_ab = dendrogram[t][2]

        objective_sum__term += p_ab * np.log(d_ab)
        objective_log_term += p_a * p_b / float(d_ab)

        graph.add_node(u)
        neighbors_a = list(graph.neighbors(a))
        neighbors_b = list(graph.neighbors(b))
        for v in neighbors_a:
            graph.add_edge(u, v, weight=graph[a][v]['weight'])
        for v in neighbors_b:
            if graph.has_edge(u, v):
                graph[u][v]['weight'] += graph[b][v]['weight']
            else:
                graph.add_edge(u, v, weight=graph[b][v]['weight'])
        graph.remove_node(a)
        graph.remove_node(b)
        w[u] = w.pop(a) + w.pop(b)
        u += 1

    return objective_sum__term + np.log(objective_log_term)
