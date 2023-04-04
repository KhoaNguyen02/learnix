import numpy as np
from utils.misc import *

class SGDRegression(object):
     def __init__(self, learning_rate=0.01, n_iters=100, lambd=0.01, regularizer=None):
          """Linear regression using gradient descent.

          Args:
              learning_rate (float, optional): learning rate of the gradient descent. Defaults to 0.01.
              n_iters (int, optional): number of iterations. Defaults to 100.
              lambd (float, optional): regularization parameter. Defaults to 0.01.
              regularizer (_type_, optional): type of regularization. Defaults to None. The options are:
                    - l1: l1 regularization
                    - l2: l2 regularization
                    - l1_l2: l1 and l2 regularization
                    - None: no regularization
          """
          self.lr = learning_rate
          self.n_iters = n_iters
          self.lambd = lambd
          assert regularizer in ['l1', 'l2', 'l1_l2', None], 'Invalid regularization'
          self.regularization = regularizer
     
     def _regularizer(self):
          """Compute the regularization term.

          Returns:
              float: regularization term.
          """
          if self.regularization == 'l1':
               return l1_regularizer(self.beta, self.lambd, grad=True)
          elif self.regularization == 'l2':
               return l2_regularizer(self.beta, self.lambd, grad=True)
          elif self.regularization == 'l1_l2':
               return l1_l2_regularizer(self.beta, self.lambd, grad=True)
          else:
               return 0
          
     def fit(self, X, y):
          """Fit the model.

          Args:
              X (np.array): of shape (n_samples, n_features) containing the training data.
              y (np.array): of shape (n_samples, ) containing the target values.
          """
          X = np.insert(X, 0, 1, axis=1)
          # init parameters
          n_samples, n_features = X.shape
          self.beta = np.zeros(n_features)
     
          # gradient descent
          for _ in range(self.n_iters):
               y_pred = np.dot(X, self.beta)
               grad = -2/n_samples * np.dot(X.T, y - y_pred)
               grad += self._regularizer()
               self.beta -= self.lr * grad

     def predict(self, X):
          """Predict using the linear model.

          Args:
              X (np.array): of shape (n_samples, n_features) containing the data to make predictions on.

          Returns:
              np.array: of shape (n_samples, ) containing the predicted values.
          """
          X = np.insert(X, 0, 1, axis=1)
          return np.dot(X, self.beta)
     
     def get_coef(self):
          """Get the coefficients of the linear model.

          Returns:
              np.array: of shape (n_features, ) containing the coefficients.
          """
          return self.beta[1:]
     
     def get_intercept(self):
          """Get the intercept of the linear model.

          Returns:
              float: intercept.
          """
          return self.beta[0]
     
class LinearRegression(SGDRegression):
     def __init__(self, learning_rate=0.01, n_iters=100, gradient_descent=False, lambd=0.01, regularizer=None):
          """Linear regression.

          Args:
              learning_rate (float, optional): learning rate of gradient descent. Defaults to 0.01.
              n_iters (int, optional): number of iterations. Defaults to 100.
              gradient_descent (bool, optional): whether to use gradient descent or not. Defaults to False.
              lambd (float, optional): regularization parameter. Defaults to 0.01.
              regularizer (_type_, optional): type of regularization. Defaults to None. The options are:
                    - l1: l1 regularization
                    - l2: l2 regularization
                    - l1_l2: l1 and l2 regularization
                    - None: no regularization
          """
          super().__init__(learning_rate, n_iters, lambd, regularizer)
          self.gradient_descent = gradient_descent

     def fit(self, X, y):
          if self.gradient_descent:
               super().fit(X, y)
          else:
               X = np.insert(X, 0, 1, axis=1)
               self.beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

class PolynomialRegression:
     def __init__(self, degree=2):
          """Polynomial regression.
          
          Args:
              degree (int, optional): degree of the polynomial. Defaults to 2.
          """
          self.degree = degree

     def _create_poly_features(self, X):
          """Create polynomial features.

          Args:
              X (np.array): of shape (n_samples, n_features) containing the data.

          Returns:
              np.array: of shape (n_samples, n_features * degree) containing the polynomial features.
          """
          n_samples, n_features = X.shape
          X_poly = np.zeros((n_samples, n_features * self.degree))
          for i in range(n_samples):
               X_poly[i] = X[i].repeat(self.degree) ** np.arange(1, self.degree + 1)
          return X_poly
     
     def fit(self, X, y):
          X_poly = self._create_poly_features(X)
          self.linear_regression = LinearRegression()
          self.linear_regression.fit(X_poly, y)

     def predict(self, X):
          X_poly = self.create_poly_features(X)
          return self.linear_regression.predict(X_poly)