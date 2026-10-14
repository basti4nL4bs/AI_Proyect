import numpy as np
import pandas as pd

# Dataset original
data = pd.DataFrame(
    {
        "precio": [50, 120, 80, 250, 90, 300, 180, 200, 70, 140, 220, 110],
        "categoria": [1, 2, 1, 3, 1, 3, 2, 3, 1, 2, 3, 2],
        "popularidad": [100, 200, 150, 350, 120, 420, 260, 300, 90, 230, 310, 180],
        "etiqueta": [
            "Electrónica",
            "Hogar",
            "Electrónica",
            "Deporte",
            "Electrónica",
            "Deporte",
            "Hogar",
            "Deporte",
            "Electrónica",
            "Hogar",
            "Deporte",
            "Hogar",
        ],
    }
)

X = data[["precio", "categoria", "popularidad"]].values
y = data["etiqueta"].values


# Implementación manual de KNN para evitar dependencias con DLLs bloqueadas por Windows
class KNNRecommender:
    def __init__(self, n_neighbors=3):
        self.k = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train)

    def predict(self, X_test):
        predictions = []
        for point in X_test:
            # Distancia euclidiana entre el punto y todos los datos de entrenamiento
            distances = np.sqrt(np.sum((self.X_train - point) ** 2, axis=1))
            # Índices de los k vecinos más cercanos
            k_indices = np.argsort(distances)[: self.k]
            k_nearest_labels = self.y_train[k_indices]
            # Voto por mayoría
            labels, counts = np.unique(k_nearest_labels, return_counts=True)
            predictions.append(labels[np.argmax(counts)])
        return np.array(predictions)


# División estratificada simple (manteniendo representatividad por etiqueta)
def stratified_split(features, labels, test_size=0.25, seed=42):
    np.random.seed(seed)
    train_idx, test_idx = [], []
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        np.random.shuffle(indices)
        n_test = max(1, int(len(indices) * test_size))
        test_idx.extend(indices[:n_test])
        train_idx.extend(indices[n_test:])
    return (
        features[train_idx],
        features[test_idx],
        labels[train_idx],
        labels[test_idx],
    )


X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.25)

model = KNNRecommender(n_neighbors=3)
model.fit(X_train, y_train)


def recommend(product_features):
    """Devuelve la etiqueta recomendada para un conjunto de características del producto."""
    features = np.asarray(product_features, dtype=float)
    if features.shape != (3,):
        raise ValueError(
            "product_features debe contener 3 valores: precio, categoria y popularidad."
        )

    prediction = model.predict(features.reshape(1, -1))[0]
    return prediction


if __name__ == "__main__":
    products = [
        [85, 1, 130],
        [190, 2, 280],
        [320, 3, 410],
    ]

    for product in products:
        print(f"Producto {product} -> Recomendación: {recommend(product)}")