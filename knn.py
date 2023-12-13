# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:39:29 2023

@author: hugop
"""
class KNNClassifier:
    def KNNbuilder(self, k=3):
        # constructor
        self.k = k  # número de vecinos
        self.X_train = []  # Lista para almacenar datos de entrenamiento
        self.y_train = []  # Lista para almacenar etiquetas de entrenamiento

    def fit(self, X_train, y_train):
        # método para entrenar el clasificador
        self.X_train = X_train  # Asigna los datos de entrenamiento
        self.y_train = y_train  # Asigna las etiquetas de entrenamiento

    def predict(self, X_test):
        # método para predecir etiquetas de datos de prueba
        predictions = [self._predict(x) for x in X_test]
        return predictions

    def _predict(self, x):
        # método para predecir la etiqueta de un solo punto
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_neighbors_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_neighbors_indices]
        most_common = self._most_common(k_nearest_labels)
        return most_common

    def _euclidean_distance(self, a, b):
        # método para calcular la distancia euclidiana entre dos puntos
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    def _most_common(self, lst):
        # método para encontrar el elemento más común en una lista
        counter = {}
        for item in lst:
            counter[item] = counter.get(item, 0) + 1
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        return most_common[0]

# EJEMPLO DE USO
if __name__ == "__main__":
    # datos de entrenamiento
    X_train = [[1, 2], [2, 3], [3, 1], [4, 2], [1, 3], [2, 2]]
    y_train = [0, 0, 1, 1, 2, 2]

    # datos de prueba
    X_test = [[2, 3.5]]

    # crear y entrenar el clasificador k-NN
    knn_classifier = KNNClassifier()
    knn_classifier.KNNbuilder(k=3)
    knn_classifier.fit(X_train, y_train)

    # realizar predicciones
    predictions = knn_classifier.predict(X_test)

    print("Predicción:", predictions)


