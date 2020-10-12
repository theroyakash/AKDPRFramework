import numpy as np

def shuffle(X, y, seed=None)
    if seed:
        np.random.seed(seed)

    index = np.arange(X.shape[0])
    np.random.shuffle(index)

    return X[index], y[index]

def batch_iterator(X, y=None, batch_size=32):
    '''
    Batch generator class

        Args:
            - X: X data
            - y: labels for each X data
            - batch_size: Batch size you want to generate. Defaults to 32
    '''
    number_of_samples = X.shape[0]

    for i in np.arange(0, number_of_samples, batch_size):
        start, end = i, min(i + batch_size, number_of_samples)
        if y is not None:
            yield X[start:end], y[start:end]
        else:
            yield X[start:end]
