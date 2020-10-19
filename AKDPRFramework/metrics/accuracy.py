import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Compare y_true to y_pred and return the accuracy
        Args:
            - ``y_true``: True labels
            - ``y_pred``: Prediction back from the model

        Returns:
            accuracy metrics
    """

    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy
