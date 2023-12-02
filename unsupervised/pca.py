import numpy as np

class PCA:
     def __init__(self, n_components):
          """Principal component analysis (PCA), which is a dimensionality reduction technique.
          Parameters
          ----------
          n_components : int
              Number of components to keep.
          """
          self.n_components = n_components
     
     def fit(self, X):
          """Fit the model with X.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features
              Training data

          Returns
          -------
          PCA
              Fitted model
          """
          self.mean = np.mean(X, axis=0)
          X = X - self.mean
          self.cov = np.cov(X.T)
          eig_vals, eig_vecs = np.linalg.eig(self.cov)
          eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
          eig_pairs.sort(key=lambda x: x[0], reverse=True)
          self.eig_pairs = eig_pairs
          self.components = np.array([x[1] for x in eig_pairs[:self.n_components]])
          return self
     
     def transform(self, X):
          """Apply dimensionality reduction on X.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features
              Training data

          Returns
          -------
          np.ndarray of shape (n_samples, n_components) where n_samples is the number of samples and n_components is the number of components
              Reduced data
          """
          X = X - self.mean
          return np.dot(X, self.components.T)
     
     def fit_transform(self, X):
          """Fit the model with X and apply dimensionality reduction on X.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features
              Training data

          Returns
          -------
          np.ndarray of shape (n_samples, n_components) where n_samples is the number of samples and n_components is the number of components
              Reduced data
          """
          self.fit(X)
          return self.transform(X)
     
     def get_eigenvalues(self):
          """Get eigenvalues of the covariance matrix.

          Returns
          -------
          np.ndarray of shape (n_components,)
              Eigenvalues
          """
          return np.array([x[0] for x in self.eig_pairs])
     
     def get_eigenvectors(self):
          """Get eigenvectors of the covariance matrix.

          Returns
          -------
          np.ndarray of shape (n_components, n_features) where n_components is the number of components and n_features is the number of features
              Eigenvectors
          """
          return np.array([x[1] for x in self.eig_pairs])
     
     def get_components(self):
          """Get all principal components.

          Returns
          -------
          np.ndarray of shape (n_components, n_features) where n_components is the number of components and n_features is the number of features
              Principal components
          """
          return self.components