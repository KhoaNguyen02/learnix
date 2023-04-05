import numpy as np
import multiprocessing as mp
from utils.metrics import *

class KNearestNeighbors:
     def __init__(self, k=5, distance='euclidean', p=2, weights='uniform', n_process='max'):
          """K-Nearest Neighbors Classifier. This is a brute force implementation of the K-Nearest Neighbors algorithm.

          Parameters
          ----------
          k : int, optional
               number of neighbors, by default 5
          distance : str, optional
               distance metric, by default 'euclidean'. The options are:

                    - 'euclidean': Euclidean distance
                    - 'manhattan': Manhattan distance
                    - 'minkowski': Minkowski distance, this comes with the parameter p
                    - 'cosine': Cosine distance
                    - 'hamming': Hamming distance
                    - 'chebyshev': Chebyshev distance
                    - 'jaccard': Jaccard distance

          p : int, optional
               parameter for the Minkowski distance, by default 2

          weights : str, optional
               weights for the neighbors, by default 'uniform'. The options are:

                    - 'uniform': uniform weights
                    - 'distance': weights are inversely proportional to the distance

          n_process : str, optional
               number of processes to use, by default 'max'. If 'max', the number of processes will be equal to the number of cores in the CPU.
          """
          self.k = k
          self.distance = distance
          self.p = p
          self.weights = weights
          self.n_process = n_process
          self._check_params()

     def _check_params(self):
          """Check the validity of the parameters."""
          assert self.k > 0, 'Invalid number of neighbors.'
          assert self.distance in ['euclidean', 'manhattan', 'minkowski', 'cosine', 'hamming', 'chebyshev', 'jaccard'], 'Invalid distance.'
          assert self.p > 0, 'Invalid p.'
          assert self.weights in ['uniform', 'distance'], 'Invalid weights.'
          assert self.n_process == 'max' or self.n_process > 0, 'Invalid number of processes.'
          if self.n_process == 'max':
               self.n_process = mp.cpu_count()

     def _get_distance(self, x, y):
          """Compute the distance between two data points.

          Parameters
          ----------
          x : np.ndarray
              indicates the first data point. 
          y : np.ndarray
              indicates the second data point.

          Returns
          -------
          float
              computed distance.
          """
          dist_dict = {'euclidean': minkowski_distance(x, y, p=2), 'manhattan': minkowski_distance(x, y, p=1),
                       'minkowski': minkowski_distance(x, y, p=self.p), 'cosine': cosine_similarity(x, y, distance=True),
                       'hamming': hamming_distance(x, y), 'chebyshev': chebyshev_distance(x, y),
                       'jacard': jaccard_similarity(x, y, distance=True)}
          return dist_dict[self.distance]
     
     def _get_nearest_neighbors(self, X, y, x):
          """Get the nearest neighbors of a data point.

          Parameters
          ----------
          X : np.ndarray
               indicates the training data.
          y : np.ndarray
               indicates the training labels.
          x : np.ndarray
               indicates the data point.

          Returns
          -------
          np.ndarray
               indices of the nearest neighbors.
          """
          distances = [self._get_distance(x, X[i]) for i in range(len(X))]
          return np.argsort(distances)[:self.k]

     def fit(self, X, y):
          """Fit the model.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
               training data.
          y : np.ndarray of shape (n_samples,).
               target values.
          """
          self.X = X
          self.y = y
          self.classes = np.unique(y)

     def predict_chunk(self, X_chunk):
          """Predict the target labels for a chunk of the test data.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
               chunk of the test data.

          Returns
          -------
          np.ndarray of shape (n_samples,)
               predicted labels.
          """
          predictions = []
          for x in X_chunk:
               neighbors = self._get_nearest_neighbors(self.X, self.y, x)
               if self.weights == 'uniform':
                    predictions.append(np.argmax(np.bincount(self.y[neighbors])))
               elif self.weights == 'distance':
                    distances = [self._get_distance(x, self.X[i]) for i in neighbors]
                    predictions.append(np.argmax(np.bincount(self.y[neighbors], weights=distances)))
          return np.array(predictions)

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
          chunks = np.array_split(X, self.n_process)
          pool = mp.Pool(self.n_process)
          results = pool.map(self.predict_chunk, chunks)
          pool.close()
          pool.join()
          return np.concatenate(results)