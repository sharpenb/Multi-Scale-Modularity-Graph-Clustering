import numpy as np
import matplotlib.pyplot as plt


def plot_results(results, range_param, label='', color='r', marker='o'):
    mean_results = np.mean(results, axis=1)
    min_results = np.mean(results, axis=1) - np.std(results, axis=1)
    max_results = np.mean(results, axis=1) + np.std(results, axis=1)
    plt.plot(range_param, mean_results, marker=marker, color=color, label=label)
    plt.fill_between(range_param, min_results, max_results, facecolor=color, interpolate=True, alpha=.2)


def save_results(results, range_param, directory, file_name):
    file = open(directory + file_name, "w")
    for i in range_param:
        file.write(str(i) + " ")
    file.write("\n")
    for result_list in results:
        for result in result_list:
            file.write(str(result) + " ")
        file.write(str("\n"))
    file.close()


def load_results(directory, file_name):
    file = open(directory + file_name, "r")
    range_param = []
    results = []
    for i, line in enumerate(file):
        if i == 0:
            range_param = map(float, line.split())
        else:
            results.append(map(float, line.split()))
    return range_param, results


def save_clusters(G, clusters, label, directory, file_name):
    clusters_sorted = sorted(clusters, key=len, reverse=True)
    file = open(directory + file_name, "w")
    for i, c in enumerate(clusters_sorted):
        c_sorted = sorted(c, key=G.degree, reverse=True)
        file.write("\n\nCluster " + str(i) + " (" + str(len(c)) +" nodes)")
        for u in c_sorted:
            file.write("\n" + str(u) + ": " + label[u])
    file.close()
