import numpy as np
from tqdm import tqdm
from terminaltables import AsciiTable


class NeuralNetworks():
    '''
    Neural Network Learning Class.
        Args:
            - loss: Loss class defined in loss module
            - optimizer: optimizer class from optim.optimizer
            - vaildation_data: Tuple containing lables and example (X,y)
    '''

    def __init__(self, loss, optimizer, vaildation_data=None):

        if vaildation_data is not None:
            X, y = vaildation_data
            self.validation_data = {'X': X,
                                    'y': y
                                }
        self.layers = []
        self.errors = {'training': [],
                       'validation': []
                    }
        self.loss_func = loss()


    def trainable(self, trainable):
        '''
        Freeze the weights of the layers of the NeuralNetwork
        Specify NeuralNetwork train or not to train
        '''
        for layer in self.layers:
            layer.trainable = trainable

    def add(self, layer):
        '''
        Add a new layer to the NeuralNetwork
        If this is not the first layer then set the input as the output of the last layer.
        '''
        if self.layers:
            # Set the output_shape from the last layers
            layer.set_input_shape(shape=self.layers[-1].output_shape())

        if hasattr(layer, 'initialize'):
            # If the layer has the attribute initialize then initialize
            layer.initialize(optimizer=self.optimizer)

    def _forward_pass(self, X, training=True):
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output, training)

        return layer_output

    def _backward_pass(self, loss_grad):
        '''
        Gradient Descent and backpropagation
        '''
        layer_reversed_order = reversed(self.layers)
        for layer in layer_reversed_order:
            loss_grad = layer.backward_pass(loss_grad)

    def train_on_batch(self, X, y):
        '''
        Gradient Descent on a single batch of data points
        '''
        y_pred = self._forward_pass(X)
        loss = np.mean(self.loss_func.loss(y, y_pred))
        acc = self.loss_func.acc(y, y_pred)
        loss_grad = self.loss_func.gradient(y, y_pred)
        # Update weights
        self._backward_pass(loss_grad=loss_grad)

        return loss, acc

    def test_on_batch(self, X, y):
        '''
        Model evaluation over a single batch of data points
        '''
        y_pred = self._forward_pass(X)
        loss = np.mean(self.loss_func.loss(y, y_pred))
        acc = self.loss_func.acc(y, y_pred)

        return loss, acc

    def fit(self, X, y, epochs, batch_size):
        '''
        Call to start the training

            Args:
                - X: X the input data points
                - y: lables for the data points
                - epochs: Specify the number of epochs
                - batch_size: Total batch size
        '''

        for _ in tqdm(range(epochs)):
            error_in_this_batch = []
            # Using batch iterator iterate over the batch
            for X_batch, y_batch in batch_iterator(X, y, batch_size=batch_size):
                loss, _ = self.train_on_batch(X_batch, y_batch)
                error_in_this_batch.append(loss)

            self.errors['training'].append(np.mean(error_in_this_batch))

            if self.validation_data is not None:
                val_loss, _ = self.test_on_batch(self.validation_data['X'], self.validation_data['y'])
                self.errors['validation'].append(val_loss)

        history = {'train_loss': self.errors['training'],
                   'val_loss': self.errors['validation']
                }

        return history

    def predict(self, X):
        '''
        use the trained model to predict
        '''
        return self._forward_pass(X, training=False)


    def sketch(self, name='Model skeleton'):
        print(AsciiTable([[name]]).table)
        print(f'Input shape: {str(self.layers[0].input_shape)}')
        table_informations = [['Layer Type', 'params', 'output shape']]
        total_param = 0

        for layer in self.layers:
            layer_name = layer.layer_name()
            params = layer.parameters()
            out_shape = layer.output_shape()
            table_informations.append([layer_name, str(params), str(out_shape)])

            total_param += params

        print(AsciiTable(table_informations).table)
        print(f'Total parameters {total_param}')
