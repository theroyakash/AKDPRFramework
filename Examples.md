# Examples
### Linear Regression
First let's create the data. I'll be making a mock dataset from sklearn library
```python
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features=1, noise=120)
```
Now import the LinearRegression library from the AKDPRFramework framework.
```python
from AKDPRFramework.mlops.Regression import LinearRegression
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
