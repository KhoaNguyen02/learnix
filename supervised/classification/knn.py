import numpy as np
import multiprocessing as mp
from utils.metrics import get_dist
import warnings

class Node:
     def __init__(self, X):
          """Node of KDTree

          Parameters
          ----------
          X : numpy.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features
               training data
          """
          self.X = X
          self.axis = None
          self.median = None
          self.left = None
          self.right = None

class InternalNode(Node):
     def __init__(self, X, left, right):
          """Internal node of KDTree

          Parameters
          ----------
          X : numpy.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features
               training data

          left : Node class
               left child of a node

          right : Node class
               tight child of a node
          """
          super().__init__(X)
          self.axis = X.shape[1] % self.X.shape[1]
          self.median = X.shape[0] // 2
          self.left = left
          self.right = right

     def _get_nearest_neighbors(self, X_test):
          """Get nearest neighbors of test data

          Parameters
          ----------
          X_test : numpy.ndarray of shape (n_features,) where n_features is the number of features
               test data

          Returns
          -------
          numpy.ndarray of shape (n_samples,) where n_samples is the number of samples
               indices of nearest neighbors
          """
          if X_test[self.axis] < self.X[self.median, self.axis]:
               return self.left._get_nearest_neighbors(X_test)
          else:
               return self.right._get_nearest_neighbors(X_test)
     
class LeafNode(Node):
     def __init__(self, X):
          """Leaf node of KDTree

          Parameters
          ----------
          X : numpy.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features
               training data
          """
          super().__init__(X)
          
     def _get_nearest_neighbors(self, *args):
          """Get nearest neighbors of test data
          
          Returns
          -------
          numpy.ndarray of shape (n_samples,) where n_samples is the number of samples
               indices of nearest neighbors
          """
          return np.arange(self.X.shape[0])
     
class KDTree:
     def __init__(self, X, dist_type='euclidean', p=2, leaf_size=30):
          """KDTree algorithm for finding nearest neighbors.

          Parameters
          ----------
          X : numpy.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features
               training data

          dist_type : str, optional
               type of metric distance used, by default 'euclidean'. The options are:

                    - 'euclidean': Euclidean distance
                    - 'manhattan': Manhattan distance
                    - 'minkowski': Minkowski distance
                    - 'cosine': Cosine distance
                    - 'chebyshev': Chebyshev distance
                    - 'hamming': Hamming distance
                    - 'jaccard': Jaccard distance

          p : int, optional
               parameter for Minkowski distance, by default 2

          leaf_size : int, optional
               maximum number of samples in a leaf node, by default 30
          """
          self.X = X
          self.dist_type = dist_type
          self.p = p
          self.leaf_size = leaf_size
          self.root = self._build_tree(X, 0)

     def _build_tree(self, X, depth):
          """Build KDTree.

          Parameters
          ----------
          X : numpy.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features
               training data

          depth : int
               depth of a node

          Returns
          -------
          Node class
               root node of KDTree
          """
          if X.shape[0] <= self.leaf_size:
               return LeafNode(X)
          else:
               axis = depth % X.shape[1]
               X = X[X[:, axis].argsort()]
               median = X.shape[0] // 2
               return InternalNode(X, self._build_tree(X[:median], depth+1), self._build_tree(X[median:], depth+1))
          
     def get_nearest_neighbors(self, X_test):
          """Get nearest neighbors of test data.

          Parameters
          ----------
          X_test : numpy.ndarray of shape (n_features,) where n_features is the number of features
               test data

          Returns
          -------
          numpy.ndarray of shape (n_samples,) where n_samples is the number of samples
               indices of nearest neighbors
          """
          return self.root._get_nearest_neighbors(X_test)

class KNN:
     def __init__(self, k=5, dist_type='euclidean', p=2, leaf_size=30, weights='uniform', algorithm='brute', n_process='max'):
          """KNN algorithm for classification

          Parameters
          ----------
          k : int, optional
               number of nearest neighbors, by default 5

          dist_type : str, optional
               type of metric distance used, by default 'euclidean'. The options are:

                    - 'euclidean': Euclidean distance
                    - 'manhattan': Manhattan distance
                    - 'minkowski': Minkowski distance
                    - 'cosine': Cosine distance
                    - 'chebyshev': Chebyshev distance
                    - 'hamming': Hamming distance
                    - 'jaccard': Jaccard distance

          p : int, optional
               parameter for Minkowski distance, by default 2

          leaf_size : int, optional
               the maximum number of samples in a leaf node, by default 30

          weights : str, optional
               weights for each neighbor, by default 'uniform'. The options are:

                    - 'uniform': Uniform weights, where all points in each neighborhood are weighted equally
                    - 'distance': Weighted by the inverse of their distance

          algorithm : str, optional
               algorithm used to compute the nearest neighbors, by default 'brute'. The options are:

                    - 'brute': Brute-force approach, where all points are compared
                    - 'kd_tree': KDTree algorithm, where the tree is built to reduce the number of comparisons

          n_process : str, optional
               number of processes to use, by default 'max'. The options are:

                    - 'max': Use all available processes of the machine
                    - 'int': Use the specified number of processes
          """
          self.k = k
          self.dist_type = dist_type
          self.p = p
          self.leaf_size = leaf_size
          self.weights = weights
          self.algorithm = algorithm
          self.n_process = n_process
          if self.n_process == 'max':
               self.n_process = mp.cpu_count()
          self._check_params()

     def _check_params(self):
          """Check the validity of parameters."""
          assert self.k > 0, 'k must be greater than 0'
          assert self.dist_type in ['euclidean', 'manhattan', 'minkowski', 'cosine', 'chebyshev', 'hamming', 'jaccard'], 'Invalid distance type'
          assert self.weights in ['uniform', 'distance'], 'Invalid weights'     
          assert self.algorithm in ['brute', 'kd_tree'], 'Invalid algorithm'
          assert self.n_process > 0, 'n_process must be greater than 0'
          assert self.p > 0, 'p must be greater than 0'
          assert self.leaf_size > 0, 'leaf_size must be greater than 0'

     def _get_nearest_neighbors(self, X, X_test):
          """Get nearest neighbors of test data.

          Parameters
          ----------
          X : numpy.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features
               training data

          X_test : numpy.ndarray of shape (n_features,) where n_features is the number of features
               test data

          Returns
          -------
          numpy.ndarray of shape (n_samples,) where n_samples is the number of samples
               indices of nearest neighbors
          """
          if self.algorithm == 'brute':
               dist = [get_dist(X_test, X[i], self.dist_type, self.p) for i in range(X.shape[0])]
               return np.argsort(dist)[:self.k]
          elif self.algorithm == 'kd_tree':
               tree = KDTree(X, self.dist_type, self.p, self.leaf_size)
               neighbors = tree.get_nearest_neighbors(X_test)
               dist = [get_dist(X_test, X[i], self.dist_type, self.p) for i in neighbors]
               return neighbors[np.argsort(dist)[:self.k]]
               
     def _predict(self, X, X_test):
          """Get the labels for obtained k nearest neighbors using majority voting.
          
          Parameters
          ----------
          X : numpy.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features
               training data

          X_test : numpy.ndarray of shape (n_features,) where n_features is the number of features
               test data
               
          Returns
          -------
          int
               predicted labels
          """
          neighbors = self._get_nearest_neighbors(X, X_test)
          if self.weights == 'uniform':
               return np.bincount(self.y[neighbors]).argmax()
          elif self.weights == 'distance':
               dist = [get_dist(X_test, X[i], self.dist_type, self.p) for i in neighbors]
               return np.bincount(self.y[neighbors], weights=1/np.array(dist)).argmax()
          
     def fit(self, X, y):
          """Fit the model.

          Parameters
          ----------
          X : numpy.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features
               training data

          y : numpy.ndarray of shape (n_samples,) where n_samples is the number of samples
               target labels
          """
          self.X = X
          self.y = y
          self.classes = np.unique(y)
          self.n_classes = len(self.classes)
     
     def predict(self, X_test):
          """Predict the labels for test data.

          Parameters
          ----------
          X_test : numpy.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features
               test data

          Returns
          -------
          numpy.ndarray of shape (n_samples,) where n_samples is the number of samples
               predicted labels
          """
          with mp.Pool(self.n_process) as pool:
               return pool.starmap(self._predict, [(self.X, X_test[i]) for i in range(X_test.shape[0])])