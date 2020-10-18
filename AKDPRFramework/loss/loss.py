import numpy as np

# Loss class
class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0

# Squard Error Loss
class SquareLoss(Loss):
    def __init__(self): 
        pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)

# Cross Entropy Loss
class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)
    
    def accuracy_score(self, y_true, y_pred):
        """
        Compare ``y_true`` to ``y_pred`` and return the accuracy
        """
        accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
        return accuracy

    def acc(self, y, p):
        return self.accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)
