def clusters_dict2clusters_list(cluster_dict):
    i = 0
    cluster_index = {}
    cluster_list = []
    for u, c in cluster_dict.items():
        if c not in cluster_index:
            cluster_list.append([u])
            cluster_index[c] = i
            i += 1
        else:
            cluster_list[cluster_index[c]].append(u)
    return cluster_list


def clusters_list2clusters_dict(cluster_list):
    cluster_dict = {}
    for i, c in enumerate(cluster_list):
        for u in c:
            cluster_dict[u] = i
    return cluster_dict
