import numpy as np
import cvxopt
import matplotlib.pyplot as plt
import multiprocessing as mp
from utils.misc import linear_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel

cvxopt.solvers.options['show_progress'] = False

class SVM:
     """Support Vector Machine classifier. Only binary classification is supported.
     
     Parameters
     ----------
     C : float, optional (default=1.0)
          Penalty parameter C of the error term.
     kernel : string, optional (default='linear')
          Specifies the kernel type to be used in the algorithm. It must be one of 'linear', 'poly', 'rbf', 'sigmoid'.
     degree : int, optional (default=3)
          Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
     gamma : float, optional (default='auto')
          Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If gamma is 'auto' then 1 / n_features will be used instead.
     coef0 : float, optional (default=1)
          Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
     """
     def __init__(self, C=1.0, kernel='linear', degree=3, gamma='auto', coef0=1):
          self.C = C
          self.kernel = kernel
          self.degree = degree
          self.gamma = gamma
          self.coef0 = coef0
          self._check_params()

     def _kernel(self, x1, x2):
          """Return the kernel function evaluated on x1 and x2.

          Parameters
          ----------
          x1 : ndarray, shape (n_features,) or (n_samples, n_features)
               vector or matrix of samples
          x2 : ndarray, shape (n_features,) or (n_samples, n_features)
               vector or matrix of samples

          Returns
          -------
          kernel_matrix : ndarray, shape (n_samples1, n_samples2) or (n_samples,) where n_samples 1 is the number of samples in x1 and n_samples2 is the number of samples in x2
               matrix of kernel evaluations
          """
          if self.kernel == 'linear':
               return linear_kernel(x1, x2)
          elif self.kernel == 'poly':
               return polynomial_kernel(x1, x2, self.degree, self.gamma, self.coef0)
          elif self.kernel == 'rbf':
               return rbf_kernel(x1, x2, self.gamma)
          elif self.kernel == 'sigmoid':
               return sigmoid_kernel(x1, x2, self.gamma, self.coef0)
     
     def _check_params(self):
          """Check the validity of the parameters."""
          assert self.C > 0, 'Invalid C.'
          assert self.kernel in ['linear', 'poly', 'rbf', 'sigmoid'], 'Invalid kernel.'
          assert self.degree > 0, 'Invalid degree.'
          assert self.gamma == 'auto' or self.gamma > 0, 'Invalid gamma.'
          assert self.coef0 >= 0, 'Invalid coef0.'
          
     def fit(self, X, y):
          """Fit the model.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
               training data.
          y : np.ndarray of shape (n_samples,).
               target values.
          """
          n_samples, n_features = X.shape

          # Check the validity training data and target labels
          self.n_classes = len(np.unique(y))
          if self.n_classes > 2:
               raise ValueError('There are '+ str(self.n_classes) + ' classes in the data. Use a multiclass SVM classifier instead.')
          if not np.all(np.unique(y) == np.array([-1, 1])):
               raise ValueError('The target labels must be -1 or 1.')

          # Set kernel coefficient
          if self.gamma is 'auto':
               self.gamma = 1 / n_features
          
          # Compute the kernel matrix
          K = np.zeros((n_samples, n_samples))
          for i in range(n_samples):
               for j in range(n_samples):
                    K[i, j] = self._kernel(X[i], X[j])
          
          # Solve the dual problem
          P = cvxopt.matrix(np.outer(y, y) * K)
          q = cvxopt.matrix(-np.ones(n_samples))
          A = cvxopt.matrix(y, (1, n_samples), 'd')
          b = cvxopt.matrix(0.0)
          G = cvxopt.matrix(np.vstack((np.diag(-np.ones(n_samples)), np.identity(n_samples))))
          h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
          solution = cvxopt.solvers.qp(P, q, G, h, A, b)
          a = np.ravel(solution['x'])
          
          # Compute the bias
          sv = a > 1e-5
          ind = np.arange(len(a))[sv]
          self.a = a[sv]
          self.sv = X[sv]
          self.sv_y = y[sv]
          self.b = 0
          for n in range(len(self.a)):
               self.b += self.sv_y[n]
               self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
          self.b /= len(self.a)
          
          # Compute the weights if linear kernel
          if self.kernel == 'linear':
               self.w = np.zeros(n_features)
               for n in range(len(self.a)):
                    self.w += self.a[n] * self.sv_y[n] * self.sv[n]
          
     
     def project(self, X):
          """Project data X into the learned hyperplane.
          
          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
               training data.
               
          Returns
          -------
          y_predict : np.ndarray of shape (n_samples,)
               predicted labels.
          """
          if self.kernel == 'linear':
               return np.dot(X, self.w) + self.b
          else:
               y_predict = np.zeros(len(X))
               for i in range(len(X)):
                    s = 0
                    for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                         s += a * sv_y * self._kernel(X[i], sv)
                    y_predict[i] = s
               return y_predict + self.b
          
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
          return np.sign(self.project(X))

class MultiClassSVM:
     """Multi-class Support Vector Machine classifier.
     
     Parameters
     ----------
     C : float, optional (default=1.0)
          Penalty parameter C of the error term.
     kernel : string, optional (default='linear')
          Specifies the kernel type to be used in the algorithm. It must be one of 'linear', 'poly', 'rbf', 'sigmoid'.
     degree : int, optional (default=3)
          Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
     gamma : float, optional (default='auto')
          Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If gamma is 'auto' then 1/n_features will be used instead.
     coef0 : float, optional (default=1)
          Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
     """
     def __init__(self, C=1.0, kernel='linear', degree=3, gamma='auto', coef0=1, n_process='max'):
          self.C = C
          self.kernel = kernel
          self.degree = degree
          self.gamma = gamma
          self.coef0 = coef0
          self.n_process = n_process
          if self.n_process == 'max':
               self.n_process = mp.cpu_count()

     def fit(self, X, y):
          """Fit the model.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
               training data.
          y : np.ndarray of shape (n_samples,).
               target values.
          """
          self.classes = np.unique(y)
          self.n_classes = len(self.classes)
          self.models = []
          with mp.Pool(self.n_process) as pool:
               for i in range(self.n_classes):
                    y_i = np.where(y == self.classes[i], 1, -1)
                    model = SVM(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0)
                    model.fit(X, y_i)
                    self.models.append(model)

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
          y_predict = np.zeros((len(X), self.n_classes))
          for i in range(self.n_classes):
               y_predict[:, i] = self.models[i].predict(X)
          return self.classes[np.argmax(y_predict, axis=1)]
     



     

     

