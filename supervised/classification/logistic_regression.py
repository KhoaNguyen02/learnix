import numpy as np
import matplotlib.pyplot as plt
from utils.misc import *
from utils.activation_functions import *

class LogisticRegression:
     def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-3, lambd=0.01, regularizer='l2', verbose=False):
          """Logistic regression classifier.

          Parameters
          ----------
          learning_rate : float, optional
               learning rate of the classifier, by default 0.01
          max_iter : int, optional
               maximum number of iterations, by default 1000
          tol : _type_, optional
               tolerance for the stopping criterion, by default 1e-3
          lambd : float, optional
               regularization parameter, by default 0.01
          regularizer : str, optional
               type of regularizer, by default 'l2'. The options are {'l1', 'l2', 'elasticnet', None}:

                    - 'l1': L1 regularization
                    - 'l2': L2 regularization
                    - 'elasticnet': combination of L1 and L2 regularization
                    - None: no regularization
          verbose : bool, optional
               whether to print the cost at each iteration, by default False
          """
          self.learning_rate = learning_rate
          self.max_iter = max_iter
          self.tol = tol
          self.lambd = lambd
          self.regularizer = regularizer
          self.verbose = verbose
          self._check_params()
     
     def _check_params(self):
          """Check the validity of the parameters."""
          assert self.learning_rate > 0, 'Invalid learning rate.'
          assert self.max_iter > 0, 'Invalid maximum number of iterations.'
          assert self.tol > 0, 'Invalid tolerance.'
          assert self.lambd > 0, 'Invalid regularization parameter.'
          assert self.regularizer in ['l1', 'l2', 'l1_l2', None], 'Invalid regularizer.'
     
     def _initialize_weights(self, n_features):
          """Initialize weights and bias.

          Parameters
          ----------
          n_features : int
              number of features in the dataset.
          """

          self.w = np.zeros(n_features)
          self.b = 0
     
     def _penalty(self):
          """Compute the penalty.

          Returns
          -------
          np.ndarray
              penalty term.
          """
          if self.regularizer == 'l1':
               return self.lambd * np.sum(np.abs(self.w))
          elif self.regularizer == 'l2':
               return self.lambd * np.sum(self.w ** 2)
          elif self.regularizer == 'l1_l2':
               return self.lambd * (np.sum(np.abs(self.w)) + np.sum(self.w ** 2))
          else:
               return 0
     
     def _forward(self, X):
          """Forward pass.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features)
              training data where n_samples is the number of samples and n_features is the number of features.

          Returns
          -------
          np.ndarray of shape (n_samples,)
               predicted values.
          """
          return sigmoid(np.dot(X, self.w) + self.b)
     
     def _backward(self, X, y, y_pred):
          """Backward pass.
          
          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features)
              training data where n_samples is the number of samples and n_features is the number of features.
          y : np.ndarray of shape (n_samples,)
              target values.
          y_pred : np.ndarray of shape (n_samples,)
               predicted values.
          """
          m = X.shape[0]
          self.w -= self.learning_rate * np.dot(X.T, (y_pred - y)) / m + self._penalty()
          self.b -= self.learning_rate * np.sum(y_pred - y) / m + self._penalty()
     
     def fit(self, X, y):
          """Fit the model.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
               training data.
          y : np.ndarray of shape (n_samples,).
               target values.
          """
          self._initialize_weights(X.shape[1])
          self.costs = []
          for i in range(self.max_iter):
               y_pred = self._forward(X)
               self._backward(X, y, y_pred)
               cost = cross_entropy_loss(y, y_pred)
               self.costs.append(cost)
               if self.verbose:
                    print(f'Iteration {i + 1}: cost = {cost}')
               if i > 0 and np.abs(cost - self.costs[-2]) < self.tol:
                    break
     
     def predict(self, X):
          """Predict the target labels for the test data.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
               test data.

          Returns
          -------
          np.ndarray of shape (n_samples,)
               predicted labels for the test data.
          """
          y_pred = self._forward(X)
          return np.round(y_pred).astype(int)