import numpy as np


def paris_local(G, node):
    F = G.copy()
    n_nodes = F.number_of_nodes()
    D = []
    w = {u: 0 for u in range(n_nodes)}
    wtot = 0
    for (u, v) in F.edges():
        weight = F[u][v]['weight']
        w[u] += weight
        w[v] += weight
        wtot += weight
        if u != v:
            wtot += weight
    s = {u: 1 for u in range(n_nodes)}

    t = 0
    while list(F.neighbors(node)) != []:
        d_min = float('inf')
        node_min = node
        new_node = n_nodes + t
        for u in F.neighbors(node):
            d_u = w[node] * w[u] / float(F[node][u]['weight']) / float(wtot)
            if d_u < d_min:
                d_min = d_u
                node_min = u

        D.append([node, node_min, d_min, s[node] + s[node_min]])
        # update graph
        F.add_node(new_node)
        neighbors_node = list(F.neighbors(node))
        neighbors_node_min = list(F.neighbors(node_min))
        for v in neighbors_node:
            F.add_edge(new_node, v, weight=F[node][v]['weight'])
        for v in neighbors_node_min:
            if F.has_edge(new_node, v):
                F[new_node][v]['weight'] += F[node_min][v]['weight']
            else:
                F.add_edge(new_node, v, weight=F[node_min][v]['weight'])
        F.remove_node(node)
        F.remove_node(node_min)
        # update weight and size
        w[n_nodes + t] = w.pop(node) + w.pop(node_min)
        s[n_nodes + t] = s.pop(node) + s.pop(node_min)
        # change cluster index
        node = new_node
        t += 1

    d_min = float('inf')
    while F.number_of_nodes() > 1:
        new_node = n_nodes + t
        if F.nodes()[0] != node:
            node_min = F.nodes()[0]
        else:
            node_min = F.nodes()[1]
        D.append([node, node_min, d_min, s[node] + s[node_min]])
        # update graph
        F.add_node(new_node)
        neighbors_node = list(F.neighbors(node))
        neighbors_node_min = list(F.neighbors(node_min))
        for v in neighbors_node:
            F.add_edge(new_node, v, weight=F[node][v]['weight'])
        for v in neighbors_node_min:
            if F.has_edge(new_node, v):
                F[new_node][v]['weight'] += F[node_min][v]['weight']
            else:
                F.add_edge(new_node, v, weight=F[node_min][v]['weight'])
        F.remove_node(node)
        F.remove_node(node_min)
        # update weight and size
        w[n_nodes + t] = w.pop(node) + w.pop(node_min)
        s[n_nodes + t] = s.pop(node) + s.pop(node_min)
        # change cluster index
        node = new_node
        t += 1

    return np.array(D)