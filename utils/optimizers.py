# Optimizer module
# Implementations of most used optimization algorithms
# Adam is implemented based on github.com/theroyakash/Adam

import numpy as np

# Adam
class Adam():
    '''
    Implementation of the Adam Optimization algorithms
    To import call `from utils import Adam` 
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

        self.w_updt = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

        return w - self.w_updt