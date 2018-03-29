from clustering_algorithms.louvain import *
from clustering_algorithms.paris import *
from dendrogram_manager.homogeneous_cut_slicer import *
from dendrogram_manager.heterogeneous_cut_slicer import *
from dendrogram_manager.distance_slicer import *
from experiments.experiments_manager import *

SAVE_PLOTS = False
results_file_name = "algorithms_comparison"

# List of tested algorihtms:
# - Louvain
# - Paris + homogenegeous cut
# - Paris + heterogeneous cut
# - Paris + distance cut + Louvain

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


algorithms = [('Louvain', louvain),
              ('Paris+homogeneous cut', paris_homogeneous),
              ('Paris+heterogeneous cut', paris_heterogeneous),
              ('Paris+louvain', paris_louvain)]

make_graphical_experiments(algorithms=algorithms, n_samples=1500, run_scikit_algorithms=False, SAVE_PLOTS=SAVE_PLOTS, results_file_name=results_file_name)
