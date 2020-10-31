from AKDPRFramework.utils.dataops import euclidean_distance
import numpy as np


class KNN:
    """
    K Nearest neighbor classifier in machine learning

        Args:
            - ``k`` (int): The number of closest neighbors.

        Examples::
            >>> from sklearn import datasets
            >>> from AKDPRFramework.utils.dataops import train_test_split, normalize
            >>> data = datasets.load_iris()
            >>> X = normalize(data.data)
            >>> y = data.target
            >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
            >>> from AKDPRFramework.mlops.knn import KNN
            >>> classifier = KNN(k=6)
            >>> predict = classifier.predict(X_test, X_train, y_train)
            >>> print(f'Accuracy is {accuracy_score(y_test, y_pred)}')
    """

    def __init__(self, k):
        self.k = k

    def vote(self, neighbor_samples):
        """
        Returns:
            - The most common class among the all neighbor samples.
        """
        count = np.bincount(neighbor_samples.astype('int'))
        return count.argmax()

    def predict(self, X_test, X_train, y_train):
        """
        Args:
            - ``X_test``
            - ``X_train``
            - ``y_train``

        Returns:
            Prediction from the KNN algorithm
        """
        pred = np.empty(X_test.shape[0])

        for i, test_samples in enumerate(X_test):
            # Sort the training samples by their distance to the test sample and get the K nearest
            idx = np.argsort([euclidean_distance(test_samples, x) for x in X_train])[:self.k]
            # Extract the labels of the K nearest neighboring training samples
            k_nearest_neighbors = np.array([y_train[i] for i in idx])
            # Label sample as the most common class label
            pred[i] = self.vote(k_nearest_neighbors)

        return pred
