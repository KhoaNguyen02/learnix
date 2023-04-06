import numpy as np

class BaseBayes:
     """Base class for Naive Bayes classifiers."""
     def _init_params(self, X, y):
          """Initialize parameters for the classifier.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
               Training vectors, where n_samples is the number of samples and n_features is the number of features
          y : np.ndarray of shape (n_samples,) where n_samples is the number of samples.
               Target labels
          """
          pass
     
     def _update_params(self, X, y, i):
          """Update parameters for the classifier.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
               Training vectors, where n_samples is the number of samples and n_features is the number of features
          y : np.ndarray of shape (n_samples,) where n_samples is the number of samples.
               Target labels
          i : int
               Index of the class
          """
          pass
     
     def _joint_log_likelihood(self, X):
          """Calculate the posterior log probability of the samples X, i.e log P(c) + log P(x|c).

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
               Training vectors, where n_samples is the number of samples and n_features is the number of features

          Returns
          -------
          np.ndarray of shape (n_samples, n_classes)
               Posterior log probability of the samples X.
          """
          pass
     
     def fit(self, X, y):
          """Fit the model.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
               Training vectors, where n_samples is the number of samples and n_features is the number of features
          y : np.ndarray of shape (n_samples,) where n_samples is the number of samples.
               Target labels
          """
          self.classes = np.unique(y)
          self.n_classes = len(self.classes)
          self.n_samples, self.n_features = X.shape
     
          self._init_params(X, y)
          self._get_prior(X, y)
     
          for i in range(self.n_classes):
               self._update_params(X, y, i)

     def _get_prior(self, X, y):
          """Calculate the prior probability of each class.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
               Training vectors, where n_samples is the number of samples and n_features is the number of features
          y : np.ndarray of shape (n_samples,) where n_samples is the number of samples.
               Target labels
          """
          self.class_count_ = np.zeros(self.n_classes)
          for i in range(self.n_classes):
               self.class_count_[i] = np.sum(y == self.classes[i])
          self.class_prior_ = self.class_count_ / self.n_samples
     
     def predict_log_proba(self, X):
          """Log-probability estimates for the test data X.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
               Training vectors, where n_samples is the number of samples and n_features is the number of features

          Returns
          -------
          np.ndarray of shape (n_samples, n_classes)
               Returns the log-probability of the samples for each class in the model. The columns correspond to the classes in sorted order, as they appear in the attribute classes.
          """
          log_likelihood = np.zeros((X.shape[0], self.n_classes))
          for i in range(self.n_classes):
               log_likelihood[:, i] = self._joint_log_likelihood(X, i)
          return log_likelihood
     
     def predict_proba(self, X):
          """Probability estimates for the test data X.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
               Training vectors, where n_samples is the number of samples and n_features is the number of features

          Returns
          -------
          np.ndarray of shape (n_samples, n_classes)
               Returns the probability of the samples for each class in the model. The columns correspond to the classes in sorted order, as they appear in the attribute classes.
          """
          return np.exp(self.predict_log_proba(X))
     
     def predict(self, X):
          """Predict labels for test data X.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
               Training vectors, where n_samples is the number of samples and n_features is the number of features

          Returns
          -------
          np.ndarray of shape (n_samples,)
               Returns the predicted labels for X.
          """
          return self.classes[np.argmax(self.predict_proba(X), axis=1)]
     

class GaussianNB(BaseBayes):
     def __init__(self, epsilon=1e-9):
          """Gaussian Naive Bayes classifier.

          Parameters
          ----------
          var_smoothing : float, optional (default=1e-9)
              parameter for Laplace smoothing
          """
          self.epsilon = epsilon
     
     def _init_params(self, X, y):
          self.theta = np.zeros((self.n_classes, self.n_features))
          self.sigma = np.zeros((self.n_classes, self.n_features))
     
     def _update_params(self, X, y, i):
          X_i = X[y == self.classes[i]]
          self.theta[i] = np.mean(X_i, axis=0)
          self.sigma[i] = np.var(X_i, axis=0)
          self.class_count_[i] = X_i.shape[0]
     
     def _joint_log_likelihood(self, X, i):
          jointi = np.log(self.class_count_[i] / self.n_samples)
          n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma[i]))
          n_ij -= 0.5 * np.sum(((X - self.theta[i]) ** 2) / (self.sigma[i] + self.epsilon), 1)
          return jointi + n_ij






     



