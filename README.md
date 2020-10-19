# AKDPRFramework

AKDPRFramework is a framework for deep learning research. It's currently under development. All the code is written in `numpy` from scratch. Open a new PR/ issue to start contributing.
### Docs
See the docs [here](https://akdpr.netlify.app)
### Website
You can also visit the website for this open source project [here](https://bit.ly/AKDPRFramework)

## Setup
- First download/clone this repo like `git clone https://github.com/theroyakash/AKDPRFramework.git`
- Now uninstall if any previous version installed `pip uninstall AKDPRFramework`
- Now install fresh on your machine `pip install -e AKDPRFramework`

### Alternate installation
Slower install time.
```bash
pip install https://github.com/theroyakash/AKDPRFramework/tarball/main
```

## First code
Now to check whether your installation is completed without error import AKDPRFramework
```python
import AKDPRFramework as framework
```
### Check the version
```python
print('AKDPRFramework Version is --> ' + framework.__version__)
```
## Example code
```python
from AKDPRFramework.dl.activations import Sigmoid
import numpy as np

z = np.array([0.1, 0.4, 0.7, 1])
sigmoid = Sigmoid()
return_data = sigmoid(z)

print(return_data)          # -> array([0.52497919, 0.59868766, 0.66818777, 0.73105858])
print(sigmoid.gradient(z))  # -> array([0.24937604, 0.24026075, 0.22171287, 0.19661193])
```
