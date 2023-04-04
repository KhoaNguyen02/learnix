import numpy as np
import multiprocessing as mp

class SGDRegression(object):
     def __init__(self, learning_rate=0.01, n_iters=100, lambd=0.01, regularization=None):
          self.lr = learning_rate
          self.n_iters = n_iters
          self.lambd = lambd
          assert regularization in ['l1', 'l2', 'l1_l2', None], 'Invalid regularization'
          self.regularization = regularization
     
     def _regularizer(self):
          if self.regularization == 'l1':
               return self.lambd * np.sign(self.beta)
          elif self.regularization == 'l2':
               return 2 * self.lambd* self.beta
          elif self.regularization == 'l1_l2':
               return self.lambd * np.sign(self.beta) + 2 * self.lambd* self.beta
          else:
               return 0
          
     def fit(self, X, y):
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
          X = np.insert(X, 0, 1, axis=1)
          return np.dot(X, self.beta)
     
     def get_coef(self):
          return self.beta[1:]
     
     def get_intercept(self):
          return self.beta[0]
     
class LinearRegression(SGDRegression):
     def __init__(self, learning_rate=0.01, n_iters=100, gradient_descent=False, lambd=0.01, regularization=None):
          super().__init__(learning_rate, n_iters, lambd, regularization)
          self.gradient_descent = gradient_descent

     def fit(self, X, y):
          if self.gradient_descent:
               super().fit(X, y)
          else:
               X = np.insert(X, 0, 1, axis=1)
               self.beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

class PolynomialRegression:
     def __init__(self, degree=2):
          self.degree = degree

     def _create_poly_features(self, X):
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