# Optimizer module
# Implementations of most used optimization algorithms
# Adam is implemented based on github.com/theroyakash/Adam

import numpy as np


class StochasticGradientDescent():
    def __init__(self, learning_rate=0.01, momentum=0):
        self.learning_rate = learning_rate 
        self.momentum = momentum
        self.w_update = None

    def update(self, w, gradient_wrt_w):
        # If not initialized
        if self.w_update is None:
            self.w_update = np.zeros(np.shape(w))
        # Use momentum if set
        self.w_update = self.momentum * self.w_update + (1 - self.momentum) * gradient_wrt_w
        # Move against the gradient to minimize loss
        return w - self.learning_rate * self.w_update


# Adam
class Adam():
    '''
    Implementation of the Adam Optimization algorithms
    To import call `from AKDPRFramework.optim.optimizers import Adam` 
    '''
    def __init__(self, learning_rate = 1e-3, epsilon=1e-8, b1=0.9, b2=0.999):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.m = None
        self.v = None
        # Decay rates beta 1 and beta 2
        self.b1 = b1
        self.b2 = b2

    def update(self, w, gradient_wrt_w):
        # If not initialized then initialize
        if self.m is None:
            self.m = np.zeros(np.shape(gradient_wrt_w))
            self.v = np.zeros(np.shape(gradient_wrt_w))
        
        self.m = self.b1 * self.m + (1 - self.b1) * gradient_wrt_w
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(gradient_wrt_w, 2)

        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)

        self.w_update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return w - self.w_update


class RMSprop():
    def __init__(self, learning_rate=0.01, rho=0.9):
        self.learning_rate = learning_rate
        self.Eg = None # Running average of the square gradients at w
        self.epsilon = 1e-8
        self.rho = rho

    def update(self, w, gradient_wrt_w):
        # If not initialized
        if self.Eg is None:
            self.Eg = np.zeros(np.shape(gradient_wrt_w))

        self.Eg = self.rho * self.Eg + (1 - self.rho) * np.power(gradient_wrt_w, 2)

        # Divide the learning rate for a weight by a running average of the magnitudes of recent
        # gradients for that weight
        return w - self.learning_rate *  gradient_wrt_w / np.sqrt(self.Eg + self.epsilon)