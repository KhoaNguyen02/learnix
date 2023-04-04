import numpy as np

def l1_regularizer(x, lambd, grad=False):
     """L1 regularizer.

     Args:
         x (np.array): of shape (n_features, ) containing the parameters.
         lambd (float): regularization strength.
         grad (bool, optional): whether to return the gradient or the regularizer. Defaults to False.

     Returns:
         float: regularizer or gradient of the regularizer.
     """
     if grad:
          return lambd * np.sign(x)
     else:
          return lambd * np.sum(np.abs(x))
    
def l2_regularizer(x, lambd, grad=False):
     """L2 regularizer.

     Args:
         x (np.array): of shape (n_features, ) containing the parameters.
         lambd (float): regularization strength.
         grad (bool, optional): whether to return the gradient or the regularizer. Defaults to False.

     Returns:
         float: regularizer or gradient of the regularizer.
     """
     if grad:
          return 2 * lambd * x
     else:
          return lambd * np.sum(x ** 2)
     
def l1_l2_regularizer(x, lambd, grad=False):
     """Combined L1 and L2 regularizer.

     Args:
         x (np.array): of shape (n_features, ) containing the parameters.
         lambd (float): regularization strength.
         grad (bool, optional): whether to return the gradient or the regularizer. Defaults to False.

     Returns:
         float: regularizer or gradient of the regularizer.
     """
     if grad:
          return l1_regularizer(x, lambd, grad=True) + l2_regularizer(x, lambd, grad=True)
     else:
          return l1_regularizer(x, lambd) + l2_regularizer(x, lambd)