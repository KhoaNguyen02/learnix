import numpy as np

class DBSCAN:
     def __init__(self, eps=0.5, min_samples=5):
          """Density-based spatial clustering of applications with noise (DBSCAN).
          Parameters
          ----------
          eps : float
              The maximum distance between two samples for them to be considered as in the same neighborhood.
          min_samples : int
              The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
          """
          self.eps = eps
          self.min_samples = min_samples
     
     def fit(self, X):
          """Fit the model with X.
 
          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features
              Training data
 
          Returns
          -------
          DBSCAN
              Fitted model
          """
          self.X = X
          self.labels = np.zeros(X.shape[0])
          self.cluster = 0
          for i in range(X.shape[0]):
               if self.labels[i] != 0:
                    continue
               neighbors = self._region_query(i)
               if len(neighbors) < self.min_samples:
                    self.labels[i] = -1
                    continue
               self.cluster += 1
               self.labels[i] = self.cluster
               for j in neighbors:
                    if self.labels[j] == -1:
                         self.labels[j] = self.cluster
                    if self.labels[j] != 0:
                         continue
                    self.labels[j] = self.cluster
                    neighbors2 = self._region_query(j)
                    if len(neighbors2) >= self.min_samples:
                         neighbors += neighbors2
          return self
     
     def _region_query(self, i):
          """Find all points in the neighborhood of point i.

          Parameters
          ----------
          i : int
              Index of the point to find neighbors

          Returns
          -------
          list of int
              indices of the points in the neighborhood of point i
          """
          neighbors = []
          for j in range(self.X.shape[0]):
               if np.linalg.norm(self.X[i] - self.X[j]) < self.eps:
                    neighbors.append(j)
          return neighbors
     
     def predict(self, X):
          """Apply clustering on X.
 
          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features
              Training data
 
          Returns
          -------
          np.ndarray of shape (n_samples,) where n_samples is the number of samples
              Predicted cluster labels
          """
          self.fit(X)
          return self.labels
     


     
