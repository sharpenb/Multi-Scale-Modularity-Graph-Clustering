from clustering_algorithms.spectral_clustering import *
from clustering_algorithms.louvain import *
from clustering_algorithms.paris import *
from dendrogram_manager.homogeneous_cut_slicer import *
from dendrogram_manager.heterogeneous_cut_slicer import *
from dendrogram_manager.distance_slicer import *
from experiments.running_times_experiments_manager import *

SAVE_PLOTS = True
LOAD_RESULTS = False
SAVE_RESULTS = True
directory_results = "/home/sharp/Documents/Graphs/Graph_Clustering/Results/"
results_file_name = "running_times"
n_samples = 10

# List of tested algorihtms:
# - Spectral clustering
# - Louvain
# - Paris + homogenegeous cut
# - Paris + heterogeneous cut
# - Paris + distance cut + Louvain
# - Paris + distance cut + Louvain + spectral


def paris_homogeneous(G):
    D = paris(G)
    best_cut, best_score = best_homogeneous_cut(D)
    clusters = clustering_from_homogeneous_cut(D, best_cut)
    return clusters


def paris_heterogeneous(G):
    D = paris(G)
    best_cut, best_score = best_heterogeneous_cut(D)
    clusters = clustering_from_heterogeneous_cut(D, best_cut)
    return clusters


def paris_louvain(G):
    D = paris(G)
    best_dist, best_score = best_distance(D)
    clusters = louvain(G, 1/float(best_dist))
    return clusters



algorithms = [('Spectral Clustering - (25)', lambda G: spectral_clustering(G, n_clusters=25)),
              ('Spectral Clustering - (50)', lambda G: spectral_clustering(G, n_clusters=50)),
              ('Spectral Clustering - (75)', lambda G: spectral_clustering(G, n_clusters=75)),
              ('Louvain', louvain),
              ('Paris', paris)]


print('Experiment: number of nodes')
make_n_nodes_running_times_experiment(algorithms, range_n_nodes=range(100, 1001, 50), p_edge=.1, n_samples=10,
                              SAVE_PLOTS=SAVE_PLOTS, LOAD_RESULTS=LOAD_RESULTS, SAVE_RESULTS=SAVE_RESULTS, directory_results=directory_results, results_file_name=results_file_name)

print('Experiment: probability of edge')
make_p_edge_running_times_experiment(algorithms, range_p_edge=np.linspace(.01,.99, num=10), n_nodes=500, n_samples=10,
                              SAVE_PLOTS=SAVE_PLOTS, LOAD_RESULTS=LOAD_RESULTS, SAVE_RESULTS=SAVE_RESULTS, directory_results=directory_results, results_file_name=results_file_name)
