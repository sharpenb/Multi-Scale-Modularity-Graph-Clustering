import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd


def plot_graph(G, pos, figsize=(15, 8), node_size=50, alpha =.2, nodes_numbering=False, edges_numbering=False, file_name=""):
    plt.figure(figsize=figsize)
    plt.axis('off')

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='w')
    nodes.set_edgecolor('k')
    nx.draw_networkx_edges(G, pos, alpha=alpha)

    if edges_numbering:
        w = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=w)

    if nodes_numbering:
        nx.draw_networkx_labels(G, pos)

    if file_name != "":
        plt.savefig(file_name + ".pdf", bbox_inches='tight')
        plt.savefig(file_name + ".png", bbox_inches ='tight')
    else:
        plt.show()


def plot_graph_clustering(G, clusters, pos, figsize=(15, 8), node_size=50, alpha=.2, title=True, nodes_numbering=False, edges_numbering=False, file_name=""):
    colors = ['b', 'g', 'r', 'c', 'm', 'y'] + sorted(list(mcd.XKCD_COLORS))
    k = min(len(colors), len(clusters))
    clusters = sorted(clusters, key=len, reverse=True)

    plt.figure(figsize=figsize)
    plt.axis('off')

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='w')
    nodes.set_edgecolor('k')
    nx.draw_networkx_edges(G, pos, alpha=alpha)
    for l in range(k):
        nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, nodelist=clusters[l], node_color=colors[l])
        nodes.set_edgecolor('k')

    if title:
        plt.title("(" + str(len(clusters)) + " clusters)")

    if edges_numbering:
        w = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=w)

    if nodes_numbering:
        nx.draw_networkx_labels(G, pos)

    if file_name != "":
        plt.savefig(file_name + ".pdf", bbox_inches ='tight')
        plt.savefig(file_name + ".png", bbox_inches ='tight')
    else:
        plt.show()
