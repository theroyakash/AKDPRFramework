# Examples
### Linear Regression
First let's create the data. I'll be making a mock dataset from sklearn library
```python
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features=1, noise=120)
```
Now import the LinearRegression library from the AKDPRFramework framework.
```python
from AKDPRFramework.mlops.reg import LinearRegression
```
Now make a model for the LinearRegression, and call the `model.fit()` method to start training.
```python
model = LinearRegression(iterations=100)
model.fit(X, y)
```
Paste the following code to see the error go down as we train.
```python
import matplotlib.pyplot as plt
lenght = len(model.training_errors)
plt.plot(range(lenght), model.training_errors, label='Traning Errors')
plt.title("Error Plot")
plt.ylabel('MSE')
plt.xlabel('iterations')
plt.show()
```

### Linear Regression on sklearn diabetes dataset
- First import all the necessary libraries
```python
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from AKDPRFramework.mlops.reg import LinearRegression
```
- Now load the dataset
```python
diabetes_X, diabetes_y = sklearn.datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]
```
- Now split the dataset into testing and training
```python
# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]
```

- Now define the LinearRegression Model
```python
model = LinearRegression(iterations=15)
model.fit(diabetes_X_train, diabetes_y_train)
```
- Now plot the errors.
```python
lenght = len(model.training_errors)
plt.plot(range(lenght), model.training_errors, label='Traning Errors')
plt.title("Error Plot")
plt.ylabel('MSE')
plt.xlabel('iterations')
plt.show()
```
- To get the final error type out the following code:
```python
print(f'Final error is {model.training_errors[-1]}')
```
- Predict on the test dataset
```python
pred = model.predict(diabetes_X_test)
```
- Find out the validation/test dataset loss
```python
from AKDPRFramework.loss import loss
loss = loss.SquareLoss()
loss = loss.loss(pred, diabetes_y_test)
print(f'Final validation loss is: {np.mean(loss)}')
```
Also you can use numpy to predict the mean of MSE loss like this:
```python
np.mean(0.5 * (diabetes_y_test - pred) ** 2)
```
