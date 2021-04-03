# Model Documentation

## Linear Regression
A least-squares linear regression model.

### Initialization
```python
from models.LinearRegression import LinearRegression
import numpy as np

model = LinearRegression()
```
You can regularize the model using L1 or L2 regularization:
```python
model = LinearRegression(regularization='L2')  # or 'L1'
```

### Training
`LinearRegression.train` takes the following parameters:
- `X`: `np.array`; training inputs
- `y`: `np.array`; training targets
- `lambd`: Optional, `float`; regularization parameter lambda, default 0.1
- `epochs`: Optional, `int`; number of epochs to train for, default 100
- `lr`: Optional, `float`; learning rate, default 0.001
- `val_X`: Optional, `np.array`; validation inputs
- `val_y`: Optional, `np.array`; validation targets
- `init_scale`: Optional, `float`; scaling factor for initializing weights and bias

```python
x = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
y = np.array([1, 2, 3])

model = LinearRegression()
model.train(x, y)
```

The model stores training and validation (if validation data is provided) losses through time.
You can access them directly through `model.train_errors` and `model.val_errors` respectively.
You can also plot the losses through time by calling `model.plot_training()`.

The weights and bias of the model are accessible through `model.weights` and `model.bias`.

### Other Methods
`LinearRegression.predict(X, params=None)`:
Make predictions using the model.
- `X`: `np.array`; inputs to make predictions for
- `params`: Optional, `np.array`: weights and bias of the model. If `None`, the weights and bias stored in the model are used.
