import numpy as np

def get_dist(x, y, type='euclidean', p=2):
     dist_dict = {'euclidean': minkowski_distance(x, y, p=2), 'manhattan': minkowski_distance(x, y, p=1),
                  'minkowski': minkowski_distance(x, y, p=p), 'cosine': cosine_similarity(x, y, distance=True),
                    'hamming': hamming_distance(x, y), 'chebyshev': chebyshev_distance(x, y),
                    'jaccard': jaccard_similarity(x, y, distance=True)}
     return dist_dict[type]

def minkowski_distance(x, y, p=2):
     return np.sum(np.abs(x - y) ** p) ** (1 / p)

def cosine_similarity(x, y, distance=False):
     if distance:
          return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
     else:
          return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
     
def jaccard_similarity(x, y, distance=False):
     intersection = np.sum(np.minimum(x, y))
     union = np.sum(np.maximum(x, y))
     if distance:
          return 1 - intersection / union
     else:
          return intersection / union
     
def chebyshev_distance(x, y):
     return np.max(np.abs(x - y))
     
def hamming_distance(x, y):
     return np.sum(np.abs(x - y)) / len(x)


    