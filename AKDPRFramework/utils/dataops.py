import numpy as np


def shuffle(X, y, seed=None):
    """
    Shuffles the batch from a given set of datapoints and lables.
    """
    if seed:
        np.random.seed(seed)

    index = np.arange(X.shape[0])
    np.random.shuffle(index)

    return X[index], y[index]

def batch_iterator(X, y=None, batch_size=32):
    """
    Batch generator class

        Args:
            - X: X data
            - y: labels for each X data
            - batch_size: Batch size you want to generate. Defaults to 32
    """
    number_of_samples = X.shape[0]

    for i in np.arange(0, number_of_samples, batch_size):
        start, end = i, min(i + batch_size, number_of_samples)
        if y is not None:
            yield X[start:end], y[start:end]
        else:
            yield X[start:end]


def to_categorical(x, n_col=None):
    """
    Preforms one hot encodings for the data labels

        Args:
            - ``X``: Numpy Array containing your data points
            - ``n_col``: Number of coloum for your data. If not explicitly mentioned, it's automatically calculated.

        Example::
            >>> import numpy as np
            >>> def to_categorical(x, n_col=None):
            >>>     if not n_col:
            >>>         n_col = np.amax(x) + 1

            >>>     one_hot = np.zeros((x.shape[0], n_col))
            >>>     one_hot[np.arange(x.shape[0]), x] = 1
            >>>     return one_hot

            >>> x = np.array([2, 3, 4, 1, 2, 3])
            >>> z = to_categorical(x, 5)
            >>> print(z)

            >>> x = np.array([1, 2, 3, 4, 6])
            >>> z = to_categorical(x, 7)
            >>> print(z)
    """
    if not n_col:
        n_col = np.amax(x) + 1
    
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


def diag(x):
    """
    Vector to diagonal matrix conversion.
    """
    diagonal = np.zeros((len(x), len(x)))
    for i in range(len(diagonal[0])):
        diagonal[i, i] = x[i]

    return diagonal
