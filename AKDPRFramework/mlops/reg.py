from tqdm import tqdm
import numpy as np
import math

from AKDPRFramework.dl.activations import Sigmoid
from AKDPRFramework.utils.dataops import diag


class L1():
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.linalg.norm(w)

    def grad(self, w):
        return self.alpha * np.sign(w)


class L2():
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * 0.5 * w.T.dot(w)

    def grad(self, w):
        return self.alpha * w


class L1L2():
    def __init__(self, alpha, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        l1_contr = self.l1_ratio * np.linalg.norm(w)
        l2_contr = (1 - self.l1_ratio) * 0.5 * w.T.dot(w)
        return self.alpha * (l1_contr + l2_contr)

    def grad(self, w):
        l1_contr = self.l1_ratio * np.sign(w)
        l2_contr = (1 - self.l1_ratio) * w
        return self.alpha * (l1_contr + l2_contr)


class Regression(object):
    def __init__(self, iterations, learning_rate):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.regularization = lambda x: 0

    def random_initialization(self, number_of_features):
        '''
        Random initialization of weights.
        '''
        lim = 1 / math.sqrt(number_of_features)
        self.w = np.random.uniform(-lim, lim, (number_of_features,))

    def fit(self, X, y):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []        # Record of MSE errors during training
        self.random_initialization(number_of_features=X.shape[1])

        for _ in tqdm(range(self.iterations)):
            y_pred = X.dot(self.w)
            # MSE
            mse = np.mean(0.5 * (y - y_pred) ** 2 + self.regularization(self.w))
            self.training_errors.append(mse)
            # Gradient Calculation
            grad_w = -(y - y_pred).dot(X) + self.regularization.grad(self.w)
            # Update the weights
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred

    def get_weights(self):
        """
        Returns weights at the end of training.

        Examples::
            >>> model = LinearRegression()
            >>> weights_before_training = model.get_weights()     # Random Initialization
            >>> model.fit(X, y)
            >>> weights = model.get_weights() # Updated weights after training
        """
        return self.w


class LinearRegression(Regression):
    """
    Linear Regression model
    
        Args:
            - ``iterations``: The number of training iterations the algorithm will tune the weights for.
            - ``learning_rate``: The step length that will be used when updating the weights.
            - ``gradient_descent``: True or False depending if gradient descent should be used when training. If False then we use batch optimization by least squares.
    """

    def __init__(self, iterations=1000, learning_rate=1e-3, gradient_descent=True):
        super(LinearRegression, self).__init__(iterations=iterations,
                                               learning_rate=learning_rate)
        self.gradient_descent = gradient_descent
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0

    def fit(self, X, y):
        # If not gradient descent => Least squares approximation of w
        if not self.gradient_descent:
            # Insert constant ones for bias weights
            X = np.insert(X, 0, 1, axis=1)
            # Calculate weights by least squares (using Moore-Penrose pseudoinverse)
            U, S, V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.w = X_sq_reg_inv.dot(X.T).dot(y)
        else:
            super(LinearRegression, self).fit(X, y)


class LogisticRegression:
    """
    Logistic Regression Class

        Args:
            - ``learning_rate``: Mention the learning Rate for the training. Defaults to 1e-3
            - ``gradient_descent``: True or False depending upon training with gradient or batch optimization by least squares.
    """

    def __init__(self, learning_rate=1e-3, gradient_descent=True):
        self.params = None
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.sigmoid = Sigmoid()

    def _initialize_parameters(self, X):
        number_of_features = np.shape(X)[1]
        lim = 1 / math.sqrt(number_of_features)

        self.params = np.random.uniform(-lim, lim, (number_of_features,))

    def fit(self, X, y, n_iterations=1000):
        self._initialize_parameters(X)
        # Tune parameters for n iterations
        for _ in tqdm(range(n_iterations)):
            # Make a new prediction
            y_pred = self.sigmoid(X.dot(self.params))
            if self.gradient_descent:
                # Move against the gradient of the loss function with
                # respect to the parameters to minimize the loss
                self.params -= self.learning_rate * -(y - y_pred).dot(X)
            else:
                # Make a diagonal matrix of the sigmoid gradient column vector
                diag_gradient = diag(self.sigmoid.gradient(X.dot(self.params)))
                # Batch opt:
                self.param = np.linalg.pinv(X.T.dot(diag_gradient).dot(X)).dot(X.T).dot(diag_gradient.dot(X).dot(self.param) + y - y_pred)

    def predict(self, X):
        y_pred = np.round(self.sigmoid(X.dot(self.param))).astype(int)
        return y_pred
