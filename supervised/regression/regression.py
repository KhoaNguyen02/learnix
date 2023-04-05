import numpy as np
from utils.misc import *

class SGDRegression(object):
     def __init__(self, learning_rate=0.01, n_iters=100, lambd=0.01, regularizer=None):
          """Linear regression using gradient descent.

          Parameters
          ----------
          learning_rate : float, optional
               learning rate of gradient descent, by default 0.01
          n_iters : int, optional
               number of iterations, by default 100
          lambd : float, optional
               regularization parameter, by default 0.01
          regularizer : str, optional
               type of regularizer, by default None. The options are {'l1', 'l2', 'elasticnet', None}:

                    - 'l1': L1 regularization
                    - 'l2': L2 regularization
                    - 'elasticnet': combination of L1 and L2 regularization
                    - None: no regularization
          """
          self.lr = learning_rate
          self.n_iters = n_iters
          self.lambd = lambd
          self.regularization = regularizer
     
     def _check_params(self):
          """Check the validity of the parameters."""
          assert self.lr > 0, 'Invalid learning rate.'
          assert self.n_iters > 0, 'Invalid number of iterations.'
          assert self.lambd > 0, 'Invalid regularization parameter.'
          assert self.regularization in ['l1', 'l2', 'elasticnet', None], 'Invalid regularizer.'

     def _penalty(self):
          """Compute the penalty.
          
          Returns
          -------
          float
               penalty term.
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

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
               training data.
          y : np.ndarray of shape (n_samples,).
               target values.
          """
          X = np.insert(X, 0, 1, axis=1)
          # init parameters
          n_samples, n_features = X.shape
          self.beta = np.zeros(n_features)
     
          # gradient descent
          for _ in range(self.n_iters):
               y_pred = np.dot(X, self.beta)
               grad = -2/n_samples * np.dot(X.T, y - y_pred)
               grad += self._penalty()
               self.beta -= self.lr * grad

     def predict(self, X):
          """Predict the target values for the test data.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
               test data.

          Returns
          -------
          np.ndarray of shape (n_samples,)
               predicted values for the test data.
          """ 
          X = np.insert(X, 0, 1, axis=1)
          return np.dot(X, self.beta)
     
     def get_coef(self):
          """Get the coefficients of the linear model.

          Returns
          -------
          np.ndarray of shape (n_features,)
               coefficients.
          """
          return self.beta[1:]
     
     def get_intercept(self):
          """Get the intercept of the linear model.

          Returns
          -------
          float
               intercept.
          """
          return self.beta[0]
     
class LinearRegression(SGDRegression):
     def __init__(self, learning_rate=0.01, n_iters=100, gradient_descent=False, lambd=0.01, regularizer=None):
          """Linear regression.

          Parameters
          ----------
          learning_rate : float, optional
               learning rate of gradient descent, by default 0.01
          n_iters : int, optional
               number of iterations, by default 100
          gradient_descent : bool, optional
               whether to use gradient descent, by default False
          lambd : float, optional
               regularization parameter, by default 0.01
          regularizer : str, optional
               type of regularizer, by default None. The options are {'l1', 'l2', 'elasticnet', None}:

                    - 'l1': L1 regularization
                    - 'l2': L2 regularization
                    - 'elasticnet': combination of L1 and L2 regularization
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

class PolynomialRegression(LinearRegression):
     def __init__(self, degree=2):
          """Polynomial regression.
          
          Parameters
          ----------
          degree : int, optional
               degree of the polynomial, by default 2
          """
          self.degree = degree
          super().__init__()
          assert self.degree > 0, 'Invalid degree.'

     def _create_poly_features(self, X):
          """Create polynomial features.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
               training data.
          
          Returns
          -------
          np.ndarray of shape (n_samples, n_features * degree)
               polynomial features of the training data.
          """
          n_samples, n_features = X.shape
          X_poly = np.zeros((n_samples, n_features * self.degree))
          for i in range(n_samples):
               X_poly[i] = X[i].repeat(self.degree) ** np.arange(1, self.degree + 1)
          return X_poly
     
     def fit(self, X, y):
          X_poly = self._create_poly_features(X)
          super().fit(X_poly, y)

     def predict(self, X):
          X_poly = self._create_poly_features(X)
          return super().predict(X_poly)