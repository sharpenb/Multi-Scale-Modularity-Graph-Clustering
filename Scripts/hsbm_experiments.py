from time import time
from models.hsbm import HSBM
from graph_manager.plot_tools import *
from dendrogram_manager.plot_tools import *
from dendrogram_manager.homogeneous_cut_slicer import *
from dendrogram_manager.heterogeneous_cut_slicer import *
from dendrogram_manager.cluster_cut_slicer import *
from dendrogram_manager.distance_slicer import *
from clustering_algorithms.louvain import *
from clustering_algorithms.paris import *
from experiments.resolution_analysis import *

DISPLAY_PLOTS = False
SAVE_PLOTS = True
directory_results = "/home/sharp/Documents/Graphs/Graph_Clustering/Results/"
results_file_name = "hsbm_unbalanced"
n_results = 8

### Generation of HSBM
alpha = .9
p_4 = 10.
p_4 = 4.
p_3 = p_4
p_2 = p_3 * alpha
p_1 = p_2 * alpha
hsbm = HSBM(range(200))
hsbm.divide_cluster([100, 100], [[p_4, p_1], [p_1, p_4]])
hsbm.next_level[0].divide_cluster([50, 50], [[p_4, p_2], [p_2, p_4]])
hsbm.next_level[1].divide_cluster([50, 50], [[p_4, p_2], [p_2, p_4]])
# hsbm = HSBM(range(800))
# hsbm.divide_cluster([400, 400], [[p_4, p_1], [p_1, p_4]])
# hsbm.next_level[0].divide_cluster([200, 200], [[p_4, p_2], [p_2, p_4]])
# hsbm.next_level[1].divide_cluster([200, 200], [[p_4, p_2], [p_2, p_4]])
# hsbm.next_level[0].next_level[0].divide_cluster([100, 100], [[p_4, p_3], [p_3, p_4]])
# hsbm.next_level[0].next_level[1].divide_cluster([100, 100], [[p_4, p_3], [p_3, p_4]])
# hsbm.next_level[1].next_level[0].divide_cluster([100, 100], [[p_4, p_3], [p_3, p_4]])
# hsbm.next_level[1].next_level[1].divide_cluster([100, 100], [[p_4, p_3], [p_3, p_4]])
G = hsbm.create_graph(distribution='Poisson')
pos = nx.spring_layout(G)


### Analysis of the modularity and the number of clusters w.r.t. resolution in Louvain and Paris
if DISPLAY_PLOTS:
    resolution_modularity(G)
    resolution_n_clusters(G)
if SAVE_PLOTS:
    resolution_modularity(G, file_name=results_file_name + "_resolution_modularity")
    resolution_n_clusters(G, file_name=results_file_name + "_resolution_n_clusters")

### Display information about the dataset
print(nx.info(G))
# plot_graph_clustering(G, hsbm.clusters_at_level(1), pos)
# plot_graph_clustering(G, hsbm.clusters_at_level(2), pos)
# plot_graph_clustering(G, hsbm.clusters_at_level(3), pos)

### Apply Paris
# time_paris = time()
# D = paris(G)
# time_paris = time() - time_paris
# print("Paris", time_paris)

# print("Best clusters")
# ranked_cuts, ranked_scores = ranking_cluster_cuts(D, lambda w, x, y: w * (np.log(x)-np.log(y)))
# for i in range(n_results):
#     print(i, ranked_scores[i])
#     C = [clustering_from_cluster_cut(D, ranked_cuts[i])]
#     if DISPLAY_PLOTS:
#         plot_graph_clustering(G, C, pos, alpha=.1)
#         plot_dendrogram_clustering(D, C)
#     if SAVE_PLOTS:
#         plot_graph_clustering(G, C, pos, alpha=.1, file_name=results_file_name + "_clusters_graph" + str(i))
#         plot_dendrogram_clustering(D, C, file_name=results_file_name + "_clusters_dendrogram" + str(i))
#
# print("Best homogeneous cuts")
# ranked_cuts, ranked_scores = ranking_homogeneous_cuts(D, lambda w, x, y: w * (np.log(x)-np.log(y)))
# for i in range(n_results):
#     print(i, ranked_scores[i])
#     C = clustering_from_homogeneous_cut(D, ranked_cuts[i])
#     if DISPLAY_PLOTS and pos != {}:
#         plot_graph_clustering(G, C, pos, alpha=.05)
#         plot_dendrogram_clustering(D, C)
#     if SAVE_PLOTS and pos != {}:
#         plot_graph_clustering(G, C, pos, alpha=.05, file_name=results_file_name + "_homogeneous_graph" + str(i))
#         plot_dendrogram_clustering(D, C, file_name=results_file_name + "_homogeneous_dendrogram" + str(i))
#
# print("Best heterogeneous cuts")
# ranked_cuts, ranked_scores = ranking_heterogeneous_cuts(D, n_results, lambda w, x, y: w * (np.log(x)-np.log(y)))
# for i in range(n_results):
#     print(i, ranked_scores[i])
#     C = clustering_from_heterogeneous_cut(D, ranked_cuts[i])
#     if DISPLAY_PLOTS and pos != {}:
#         plot_graph_clustering(G, C, pos, alpha=.05)
#         plot_dendrogram_clustering(D, C)
#     if SAVE_PLOTS and pos != {}:
#         plot_graph_clustering(G, C, pos, alpha=.05, file_name=results_file_name + "_heterogeneous_graph" + str(i))
#         plot_dendrogram_clustering(D, C, file_name=results_file_name + "_heterogeneous_dendrogram" + str(i))
#
# print("Best resolutions")
# ranked_distances, ranked_scores = ranking_distances(D, lambda w, x, y: w * (np.log(x)-np.log(y)))
# for i in range(n_results):
#     print(i, ranked_scores[i], 1/ranked_distances[i])
#     C = louvain(G, 1/float(ranked_distances[i]))
#     if DISPLAY_PLOTS and pos != {}:
#         plot_graph_clustering(G, C, pos, alpha=.05)
#         plot_dendrogram_clustering(D, C)
#     if SAVE_PLOTS and pos != {}:
#         plot_graph_clustering(G, C, pos, alpha=.05, file_name=results_file_name + "_resolution_graph" + str(i))
#         plot_dendrogram_clustering(D, C, file_name=results_file_name + "_resolution_dendrogram" + str(i))
