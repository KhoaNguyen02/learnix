import numpy as np

def sigmoid(z, grad=False):
     """Sigmoid function.

     Args:
          z (np.array): of shape (n_samples, ) containing the input.
          grad (bool, optional): whether to return the gradient or the function. Defaults to False.

     Returns:
          np.array: of shape (n_samples, ) containing the output.
     """
     if grad:
          return sigmoid(z) * (1 - sigmoid(z))
     else:
          return 1 / (1 + np.exp(-z))
    
def tanh(z, grad=False):
     """Hyperbolic tangent function.
     
     Args:
          z (np.array): of shape (n_samples, ) containing the input.
          grad (bool, optional): whether to return the gradient or the function. Defaults to False.
     
     Returns:
          np.array: of shape (n_samples, ) containing the output.
     """
     if grad:
          return 1 - tanh(z) ** 2
     else:
          return np.tanh(z)
     
def relu(z, grad=False):
     """Rectified linear unit function.
     
     Args:
          z (np.array): of shape (n_samples, ) containing the input.
          grad (bool, optional): whether to return the gradient or the function. Defaults to False.
     
     Returns:
          np.array: of shape (n_samples, ) containing the output.
     """
     if grad:
          return (z > 0).astype(float)
     else:
          return np.maximum(z, 0)
     
def leaky_relu(z, grad=False):
     """Leaky rectified linear unit function.
     
     Args:
          z (np.array): of shape (n_samples, ) containing the input.
          grad (bool, optional): whether to return the gradient or the function. Defaults to False.
     
     Returns:
          np.array: of shape (n_samples, ) containing the output.
     """
     if grad:
          return (z > 0).astype(float) + 0.01 * (z <= 0).astype(float)
     else:
          return np.maximum(z, 0.01 * z)
     
def elu(z, grad=False):
     """Exponential linear unit function.
     
     Args:
          z (np.array): of shape (n_samples, ) containing the input.
          grad (bool, optional): whether to return the gradient or the function. Defaults to False.
     
     Returns:
          np.array: of shape (n_samples, ) containing the output.
     """
     if grad:
          return (z > 0).astype(float) + (z <= 0).astype(float) * (np.exp(z) - 1)
     else:
          return np.maximum(z, 0) + np.minimum(z, 0) * (np.exp(z) - 1)
     
def softplus(z, grad=False):
     """Softplus function.
     
     Args:
          z (np.array): of shape (n_samples, ) containing the input.
          grad (bool, optional): whether to return the gradient or the function. Defaults to False.
     
     Returns:
          np.array: of shape (n_samples, ) containing the output.
     """
     if grad:
          return sigmoid(z)
     else:
          return np.log(1 + np.exp(z))
     
def softmax(z, grad=False):
     """Softmax function.
     
     Args:
          z (np.array): of shape (n_samples, n_classes) containing the input.
          grad (bool, optional): whether to return the gradient or the function. Defaults to False.
     
     Returns:
          np.array: of shape (n_samples, n_classes) containing the output.
     """
     if grad:
          return softmax(z) * (1 - softmax(z))
     else:
          return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)