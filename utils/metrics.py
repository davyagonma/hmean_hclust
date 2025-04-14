import numpy as np
from scipy.spatial.distance import euclidean


def intra_cluster_distance(data, labels, centroids):
    total = 0
    for i, x in enumerate(data):
        total += euclidean(x, centroids[labels[i]])**2
    return total / len(data)


def inter_cluster_distance(centroids):
    distances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            distances.append(euclidean(centroids[i], centroids[j]))
    return np.mean(distances)


def elbow_method(data, k_range):
    inertias = []
    from .clustering import kmeans

    for k in k_range:
        centroids, labels = kmeans(data, k)
        inertia = intra_cluster_distance(data, labels, centroids)
        inertias.append(inertia)

    return inertias
