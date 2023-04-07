import numpy as np

class KMeans:
     def __init__(self, n_clusters=8, init='random', max_iter=300, tol=1e-4, random_state=None):
          """K-Means Clustering Algorithm.

          Parameters
          ----------
          n_clusters : int, optional
               Number of clusters to generate, by default 8
          init : str, optional
               Method for initializing centroids, by default 'random'. Options are 'random' and 'k-means++':

               - 'random': Randomly select n_clusters data points from X as initial centroids
               - 'k-means++': Select the first centroid randomly from X. For each subsequent centroid, select data points with probability proportional to the squared distance from the nearest centroid that has already been selected.
          max_iter : int, optional
               Maximum number of iterations, by default 300
          tol : float, optional
               Tolerance for stopping criteria, by default 1e-4
          random_state : int, optional
               Random seed for reproducibility, by default None
          """
          self.n_clusters = n_clusters
          self.init = init
          self.max_iter = max_iter
          self.tol = tol
          self.random_state = random_state
          self._check_params()

     def _check_params(self):
          """Check the validity of the parameters."""
          assert self.n_clusters > 0, 'n_clusters must be greater than 0'
          assert self.init in ['random', 'k-means++'], 'init must be either random or k-means++'
          assert self.max_iter > 0, 'max_iter must be greater than 0'
          assert self.tol > 0, 'tol must be greater than 0'

     def _init_centroids(self, X):
          """Initialize centroids from training data.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features
               Training data

          Returns
          -------
          np.ndarray of shape (n_clusters, n_features) where n_clusters is the number of clusters and n_features is the number of features
               Initialized centroids
          """
          if self.init == 'random':
               np.random.seed(self.random_state)
               centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
          else:
               centroids = np.zeros((self.n_clusters, X.shape[1]))
               centroids[0] = X[np.random.choice(X.shape[0], 1, replace=False)]
               for i in range(1, self.n_clusters):
                    dist = np.array([np.min([np.linalg.norm(x - c) ** 2 for c in centroids[:i]]) for x in X])
                    probs = dist / dist.sum()
                    centroids[i] = X[np.random.choice(X.shape[0], 1, replace=False, p=probs)]
          return centroids
     
     def _assign_centroid(self, X, centroids):
          """Assign each data point to the closest centroid.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features
               Training data
          centroids : np.ndarray of shape (n_clusters, n_features) where n_clusters is the number of clusters and n_features is the number of features
               Current centroids

          Returns
          -------
          np.ndarray of shape (n_samples,)
               Closest centroid for each data point
          """
          dist = np.array([np.linalg.norm(X - c, axis=1) for c in centroids])
          return np.argmin(dist, axis=0)
     
     def _update_centroids(self, X, closest_centroids):
          """Update centroids by taking the mean of the assigned data points.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features
               Training data
          closest_centroids : np.ndarray of shape (n_samples,)
               Closest centroids for each data point

          Returns
          -------
          np.ndarray of shape (n_clusters, n_features) where n_clusters is the number of clusters and n_features is the number of features
               Updated centroids
          """
          return np.array([X[closest_centroids == i].mean(axis=0) for i in range(self.n_clusters)])
     
     def get_sse(self, X, closest_centroids, centroids):
          """Calculate the sum of squared errors between the data points and their assigned centroids.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features
               Training data
          closest_centroids : np.ndarray of shape (n_samples,)
               Assigned centroids for each data point
          centroids : np.ndarray of shape (n_clusters, n_features) where n_clusters is the number of clusters and n_features is the number of features
               Current centroids

          Returns
          -------
          float
               Sum of squared errors
          """
          return np.sum([np.linalg.norm(X[closest_centroids == i] - c) ** 2 for i, c in enumerate(centroids)])
     
     def fit(self, X):
          """Fit the model.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features
               Training data
          """
          self.centroids = self._init_centroids(X)
          for i in range(self.max_iter):
               closest_centroids = self._assign_centroid(X, self.centroids)
               new_centroids = self._update_centroids(X, closest_centroids)
               if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                    break
               self.centroids = new_centroids
     
     def predict(self, X):
          """Predict the closest centroid for each data point.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features
               Training data

          Returns
          -------
          np.ndarray of shape (n_samples,)
               Closest centroid for each data point
          """
          return self._assign_centroid(X, self.centroids)


class MiniBatchKMeans(KMeans):
     """Mini-batch K-Means clustering.

     Parameters
     ----------
     n_clusters : int, optional
          Number of clusters to generate, by default 8
     init : str, optional
          Method for initializing centroids, by default 'random'. Options are 'random' and 'k-means++':

          - 'random': Randomly select n_clusters data points from X as initial centroids
          - 'k-means++': Select the first centroid randomly from X. For each subsequent centroid, select data points with probability proportional to the squared distance from the nearest centroid that has already been selected.
     max_iter : int, optional
          Maximum number of iterations, by default 300
     tol : float, optional
          Tolerance for stopping criteria, by default 1e-4
     random_state : int, optional
          Random seed for reproducibility, by default None
     batch_size : int, optional
          Number of data points to use in each batch, by default 100
     """
     def __init__(self, n_clusters=8, init='random', max_iter=300, tol=1e-4, random_state=None, batch_size=100):
          super().__init__(n_clusters, init, max_iter, tol, random_state)
          self.batch_size = batch_size
          self.check_params()

     def check_params(self):
          """Check the validity of the parameters."""
          super()._check_params()
          assert self.batch_size > 0, 'batch_size must be greater than 0'

     def fit(self, X):
          """Fit the model.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features
               Training data
          """
          self.centroids = self._init_centroids(X)
          for i in range(self.max_iter):
               batch = X[np.random.choice(X.shape[0], self.batch_size, replace=False)]
               closest_centroids = self._assign_centroid(batch, self.centroids)
               new_centroids = self._update_centroids(batch, closest_centroids)
               if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                    break
               self.centroids = new_centroids

     

     


