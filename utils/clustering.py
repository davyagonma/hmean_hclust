import numpy as np
from scipy.spatial.distance import euclidean


def kmeans(data, k, max_iter=100, tol=1e-4):
    np.random.seed(42)
    n_samples, n_features = data.shape

    # Initialisation aléatoire des centroïdes
    centroids = data[np.random.choice(n_samples, k, replace=False)]
    prev_centroids = np.zeros_like(centroids)

    for i in range(max_iter):
        # Étape 1: Assignation des points aux centroïdes les plus proches
        labels = np.array([np.argmin([euclidean(x, c) for c in centroids]) for x in data])

        # Étape 2: Mise à jour des centroïdes
        for j in range(k):
            cluster_points = data[labels == j]
            if len(cluster_points) > 0:
                centroids[j] = np.mean(cluster_points, axis=0)

        # Vérification de convergence
        if np.all(np.abs(centroids - prev_centroids) < tol):
            break

        prev_centroids = centroids.copy()

    return centroids, labels


def hclust(data, linkage="single"):
    from collections import defaultdict

    n = len(data)
    clusters = {i: [i] for i in range(n)}
    distances = {(i, j): euclidean(data[i], data[j]) for i in range(n) for j in range(i + 1, n)}
    merge_steps = []

    while len(clusters) > 1:
        # Trouver les deux clusters les plus proches
        closest_pair = min(distances, key=distances.get)
        i, j = closest_pair
        merge_steps.append((clusters[i], clusters[j], distances[closest_pair]))

        # Fusion des clusters
        new_key = max(clusters.keys()) + 1
        clusters[new_key] = clusters[i] + clusters[j]
        del clusters[i], clusters[j]

        # Mettre à jour les distances
        distances = {
            (a, b): d for (a, b), d in distances.items() if i not in (a, b) and j not in (a, b)
        }

        for k in clusters:
            if k != new_key:
                dist = min(
                    euclidean(data[p1], data[p2])
                    for p1 in clusters[new_key]
                    for p2 in clusters[k]
                ) if linkage == "single" else max(
                    euclidean(data[p1], data[p2])
                    for p1 in clusters[new_key]
                    for p2 in clusters[k]
                )
                distances[(min(k, new_key), max(k, new_key))] = dist

    return merge_steps
