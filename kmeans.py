# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 03:18:41 2023

@author: hugop
"""

import random
import math

class KMeans:
    def KMbuilder(self, k=3, max_iters=100):
        self.k_value = k
        self.max_iters_value = max_iters
        self.centroids_data = None

    def fit(self, data):
        # inicializar centroides aleatorios
        self.centroids_data = random.sample(data, self.k_value)

        for _ in range(self.max_iters_value):
            # Asignar cada punto al cluster más cercano
            labels_result = self.assign_clusters(data)

            # actualizar los centroides
            new_centroids = [self.calculate_centroid([point for point, label in zip(data, labels_result) if label == i]) for i in range(self.k_value)]

            # verificar la convergencia
            if self.centroids_converged(new_centroids):
                break

            self.centroids_data = new_centroids

    def assign_clusters(self, data):
        labels_list = []
        for point in data:
            distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids_data]
            labels_list.append(distances.index(min(distances)))
        return labels_list

    def calculate_centroid(self, cluster):
        if not cluster:
            # Si no hay puntos en el cluster, mantener el antiguo centroide
            return random.choice(self.centroids_data)

        dimensions_value = len(cluster[0])
        centroid = [0] * dimensions_value

        for point in cluster:
            for i in range(dimensions_value):
                centroid[i] += point[i]

        return [x / len(cluster) for x in centroid]

    def euclidean_distance(self, point1, point2):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

    def centroids_converged(self, new_centroids):
        return all(point1 == point2 for point1, point2 in zip(self.centroids_data, new_centroids))


# USO
if __name__ == "__main__":
    # datos de entrenamiento
    data_list = [
        [1, 2], [2, 3], [3, 1], [4, 2], [8, 8],
        [9, 10], [10, 8], [5, 6], [7, 7], [2, 5],
        [3, 8], [6, 9], [8, 3], [9, 5], [5, 2],
        [4, 7], [6, 3], [7, 5], [1, 9], [3, 5],
        [2, 8], [10, 3], [4, 5], [8, 2], [7, 1]
    ]

    # crear y entrenar el clasificador KMeans
    kmeans_instance = KMeans()
    kmeans_instance.KMbuilder(k=3)
    kmeans_instance.fit(data_list)

    # resultados
    labels_result = kmeans_instance.assign_clusters(data_list)
    centroids_result = kmeans_instance.centroids_data

    print("Etiquetas finales:", labels_result)
    print("Centroides finales:", centroids_result)
