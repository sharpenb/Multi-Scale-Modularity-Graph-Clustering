from clustering_algorithms.agglomerative_clustering import *
from experiments.hierarchical_clustering_experiments_manager import *
from objective_functions.hierarchical_clustering import *

SAVE_PLOTS = False
LOAD_RESULTS = False
SAVE_RESULTS = False
directory_results = "/home/sharp/Documents/Graphs/Graph_Clustering/Results/"
results_file_name = "hsbm"
n_samples = 10

# List of tested algorihtms:
# - Single linkage
# - Complete linkage
# - Average linkage
# - Modular linkage


def single_linkage(graph):
    return agglomerative_clustering(graph, affinity='weighted', linkage='single', f=lambda l: - np.log(l), check=False)


def complete_linkage(graph):
    return agglomerative_clustering(graph, affinity='weighted', linkage='complete', f=lambda l: - np.log(l), check=False)


def average_linkage(graph):
    return agglomerative_clustering(graph, affinity='weighted', linkage='average', f=lambda l: - np.log(l), check=False)


def modular_linkage(graph):
    return agglomerative_clustering(graph, affinity='weighted', linkage='modular', f=lambda l: - np.log(l), check=False)


algorithms = [('SL', single_linkage),
              ('CL', complete_linkage),
              ('AL', average_linkage),
              ('ML', modular_linkage)]

# print('Experiment: number of levels')
# make_n_levels_experiment(algorithms, range_n_levels=range(6), decay_factor=.1, division_factor=2, core_community_size=10, p_in=10, n_samples=10, score=lambda graph, dendrogram: graph2tree_cost(graph, dendrogram),
#                          SAVE_PLOTS=SAVE_PLOTS, LOAD_RESULTS=LOAD_RESULTS, SAVE_RESULTS=SAVE_RESULTS, directory_results=directory_results, results_file_name=results_file_name)


# print('Experiment: decay factor')
# make_decay_factor_experiment(algorithms, range_decay_factor=np.linspace(.01,.3, num=20), n_levels=2, division_factor=2, core_community_size=10, p_in=1, n_samples=10, score=lambda graph, dendrogram: graph2tree_cost(graph, dendrogram),
#                          SAVE_PLOTS=SAVE_PLOTS, LOAD_RESULTS=LOAD_RESULTS, SAVE_RESULTS=SAVE_RESULTS, directory_results=directory_results, results_file_name=results_file_name)


# print('Experiment: division factor')
# make_division_factor_experiment(algorithms, range_division_factor=range(2, 10), n_levels=2, decay_factor=.1, core_community_size=10, p_in=10, n_samples=10, score=lambda graph, dendrogram: graph2tree_cost(graph, dendrogram),
#                          SAVE_PLOTS=SAVE_PLOTS, LOAD_RESULTS=LOAD_RESULTS, SAVE_RESULTS=SAVE_RESULTS, directory_results=directory_results, results_file_name=results_file_name)


print('Experiment: size of core communities')
make_core_community_size_experiment(algorithms, range_core_community_size=range(10, 51, 5), n_levels=2, decay_factor=.1, division_factor=2, p_in=10, n_samples=10, score=lambda graph, dendrogram: graph2tree_cost(graph, dendrogram),
                         SAVE_PLOTS=SAVE_PLOTS, LOAD_RESULTS=LOAD_RESULTS, SAVE_RESULTS=SAVE_RESULTS, directory_results=directory_results, results_file_name=results_file_name)
