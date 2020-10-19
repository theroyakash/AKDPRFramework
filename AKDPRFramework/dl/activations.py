import numpy as np


class Sigmoid:
    r"""Applies the element-wise function:

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + e^{-x}}


    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> from AKDPRFramework.dl.activations import Sigmoid
        >>> import numpy as np

        >>> z = np.array([0.1, 0.4, 0.7, 1])
        >>> sigmoid = Sigmoid()
        >>> return_data = sigmoid(z)

        >>> print(return_data)          # -> array([0.52497919, 0.59868766, 0.66818777, 0.73105858])
        >>> print(sigmoid.gradient(z))  # -> array([0.24937604, 0.24026075, 0.22171287, 0.19661193])

    """

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        r"""Computes Gradient of Sigmoid

        .. math::
            \frac{\partial}{\partial x} \sigma(x) = \sigma(x)* \left (  1- \sigma(x)\right)

        Args:
            x: input tensor

        Returns:
            Gradient of X
        """

        return self.__call__(x) * (1 - self.__call__(x))


# Softmax activations
class Softmax():
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)


# tanh activations
class TanH():
    def __call__(self, x):
        return 2 / (1 + np.exp(-2 * x)) - 1

    def gradient(self, x):
        return 1 - np.power(self.__call__(x), 2)


# ReLU
class ReLU:
    r"""Applies the rectified linear unit function element-wise. ReLU operation is defined as the following

    .. math::
        \text{ReLU}(x) = (x)^+ = \max(0, x)

    Usage::
        >>> from AKDPRFramework.dl.activations import ReLU
        >>> relu = ReLU()
        >>> z = np.array([0.1, -0.4, 0.7, 1])
        >>> print(relu(z))      # ---> array([0.1, 0. , 0.7, 1. ])
        >>> print(relu.gradient(z))   # ---> array([1, 0, 1, 1])
    """

    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        """
        Computes Gradient of ReLU

            Args:
                x: input tensor

            Returns:
                Gradient of X
        """

        return np.where(x >= 0, 1, 0)


# LeakyReLU
class LeakyReLU:
    r"""Applies the element-wise function:

    .. math::
        \text{LeakyReLU}(x) = \max(0, x) + \text{alpha} * \min(0, x)


    or

    .. math::
        \text{LeakyReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \text{alpha} \times x, & \text{ otherwise }
        \end{cases}

    Args:
        - alpha: Negative slope value: controls the angle of the negative slope in the :math:`-x` direction. Default: ``1e-2``

    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def gradient(self, x):
        """
        Computes Gradient of LeakyReLU

            Args:
                x: input tensor

            Returns:
                Gradient of X
        """
        return np.where(x >= 0, 1, self.alpha)


# SoftPlus
class SoftPlus():
    def __call__(self, x):
        return np.log(1 + np.exp(x))

    def gradient(self, x):
        return 1 / (1 + np.exp(-x))
