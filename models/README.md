# Model Documentation

Use the directory button at the top-left corner of this README to easily navigate to the relevant models and parts.

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
- `params`: Optional, `np.array`: weights and bias of the model. If `None`, the weights and bias stored in the model are used

## Logistic Regression

### Initialization
```python
from models.LogisticRegression import LogisticRegression
import numpy as np

model = LogisticRegression()
```
You can regularize the model using L1 or L2 regularization:
```python
model = LogisticRegression(regularization='L2')  # or 'L1'
```

### Training
`LogisticRegression.train` takes the following parameters:
- `X`: `np.array`; training inputs
- `y`: `np.array`; training targets
- `lambd`: Optional, `float`; regularization parameter lambda, default 0.1
- `epochs`: Optional, `int`; number of epochs to train for, default 100
- `lr`: Optional, `float`; learning rate, default 0.001
- `threshold`: Optional, `float`; probability threshold used for positive classification, default 0.5
- `val_X`: Optional, `np.array`; validation inputs
- `val_y`: Optional, `np.array`; validation targets
- `init_scale`: Optional, `float`; scaling factor for initializing weights and bias

```python
x = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
y = np.array([1, 0, 0])

model = LogisticRegression()
model.train(x, y)
```

The model stores training and validation (if validation data is provided) losses and accuracies through time.
You can access the losses directly through `model.train_errors` and `model.val_errors` respectively. The accuracies can be accessed through
`model.train_accs` and `model.val_accs`.
You can also plot the losses through time by calling `model.plot_training_loss()`. The accuracies can be plotted using `model.plot_training_acc`.

The weights and bias of the model are accessible through `model.weights` and `model.bias`.

### Other Methods
`LogisticRegression.predict(X, params=None)`:
Make predictions using the model.
- `X`: `np.array`; inputs to make predictions for
- `params`: Optional, `np.array`: weights and bias of the model. If `None`, the weights and bias stored in the model are used

`LogisticRegression.get_accuracy(X, y, params=None, threshold=0.5)`: Make predictions and compute the classification accuracy.
- `X`: `np.array`; inputs to make predictions for
- `y`: `np.array`; true labels
- `params`: Optional, `np.array`; Combined weights and biases of the model. If `None`, the params stored in the model are used (default)
- `threshold`: Optional, `float`; Probability threshold used for classifying positive predictions, default 0.5


## Probabilistic Matrix Factorization
A probabilistic matrix factorization model as described in
Salakhutdinov, R., Mnih, A. (2007). Probabilistic Matrix Factorization.
NIPS'07: Proceedings of the 20th International Conference on Neural Information Processing Systems, pp. 1257–1264.

To avoid complicated abstract explanations, the documentation speaks of the model as a movie rating prediction
model, which outputs a rating matrix R = UV, where U is a matrix of user latent variables, and
V is a matrix of movie latent variables.

The model requires the following data input format:
X is an n x 2 matrix where the first column corresponds to user IDs, and the second corresponds to movie IDs, i.e.
every row means the given user rated the given movie.
y is the corresponding n x 1 vector where every input contains the rating the user gave to the movie.

The model assumes that the validation and testing sets contain the same userIDs and movieIDs as the training set.
Furthermore, the IDs must be continuous integers starting from 1. That is, if there exists a user with ID 5, then
there must also exist users with IDs 1, 2, 3, and 4.

### Initialization
```python
from models.ProbMatrixFactorization import ProbMatrixFactorization as PMF
import numpy as np

model = PMF()
```
You can regularize the model using L1 or L2 regularization applied on both latent matrices:
```python
model = PMF(regularization='L2')  # or 'L1'
```

### Training
`ProbMatrixFactorization.train` takes the following parameters:
- `X`: `np.array`; n x 2 matrix of user IDs and movie IDs
- `y`: `np.array`; n x 1 vector of ratings
- `latent_size`: Optional, `int`; size of latent representation vectors, default 3
- `lambd_u`: Optional, `float`; regularization parameter lambda for user latent matrix, default 0.1
- `lambd_v`: Optional, `float`; regularization parameter lambda for movie latent matrix, default 0.1
- `epochs`: Optional, `int`; number of epochs to train for, default 100
- `lr`: Optional, `float`; learning rate, default 0.1
- `val_X`: Optional, `np.array`; validation userIDs and movieIDs
- `val_y`: Optional, `np.array`; validation ratings
- `init_scale`: Optional, `float`; scaling parameter for initializing latent matrices

```python
x = np.array([[1, 1],
              [1, 2],
              [1, 3],
              [1, 4],
              [2, 1],
              [2, 2],
              [2, 3]]) # Users 1 and 2, movies 1, 2, 3, 4
              
y = np.array([5, 1, 1, 3, 5, 2, 1]) # Ratings given

model = PMF()
model.train(x, y, latent_size=3, lr=0.1, val_X=np.array([[2, 4]]), val_y=np.array([4]))
```

The model stores training and validation (if validation data is provided) losses through time.
You can access them directly through `model.train_errors` and `model.val_errors` respectively.
You can also plot the losses through time by calling `model.plot_training()`.

When training is complete, the model stores the latent user and movie matrices in `model.latent_u` and `model.latent_v` respectively.

### Other Methods
`ProbMatrixFactorization.predict(X, latent_u=None, latent_v=None)`:
Make predictions using the model.
- `X`: `np.array`; n x 2 matrix of userIDs and movieIDs to make predictions for
- `latent_u`: Optional, `np.array`: User latent representation matrix. If `None`, the matrix stored in the model is used
- `latent_v`: Optional, `np.array`: Movie latent representation matrix. If `None`, the matrix stored in the model is used
