import numpy as np
import matplotlib.pyplot as plt

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

def cross_entropy_loss(y_true, y_pred, grad=False):
    """Cross-entropy loss function.

    Args:
        y_true (np.array): of shape (n_samples, 1) containing the true labels.
        y_pred (np.array): of shape (n_samples, 1) containing the predictions.
        grad (bool, optional): whether to return the gradient or the loss. Defaults to False.

    Returns:
        float: loss or gradient of the loss.
    """
    if grad:
        return y_pred - y_true
    else:
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
def linear_kernel(x1, x2):
    """Linear kernel.

    Args:
        x1 (np.array): of shape (n_features, ) containing the first sample.
        x2 (np.array): of shape (n_features, ) containing the second sample.

    Returns:
        float: the kernel value.
    """
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, p=2, gamma=1, coef=0):
    """Polynomial kernel.

    Args:
        x1 (np.array): of shape (n_features, ) containing the first sample.
        x2 (np.array): of shape (n_features, ) containing the second sample.
        p (int, optional): degree of the polynomial. Defaults to 2.
        gamma (float, optional): kernel coefficient. Defaults to 1.
        coef (float, optional): independent term. Defaults to 0.

    Returns:
        float: the kernel value.
    """
    return (gamma * np.dot(x1, x2) + coef) ** p

def rbf_kernel(x1, x2, gamma=1):
    """Radial basis function kernel.

    Args:
        x1 (np.array): of shape (n_features, ) containing the first sample.
        x2 (np.array): of shape (n_features, ) containing the second sample.
        gamma (float, optional): kernel coefficient. Defaults to 1.

    Returns:
        float: the kernel value.
    """
    return np.exp(-gamma * np.sum((x1 - x2) ** 2))

def sigmoid_kernel(x1, x2, gamma=1, coef=0):
    """Sigmoid kernel.

    Args:
        x1 (np.array): of shape (n_features, ) containing the first sample.
        x2 (np.array): of shape (n_features, ) containing the second sample.
        gamma (float, optional): kernel coefficient. Defaults to 1.
        coef (float, optional): independent term. Defaults to 0.

    Returns:
        float: the kernel value.
    """
    return np.tanh(gamma * np.dot(x1, x2) + coef)

def train_test_split(X, y, test_size=0.2, shuffle=True, seed=None):
    """Split the data into train and test sets.

    Args:
        X (np.array): of shape (n_samples, n_features) containing the features.
        y (np.array): of shape (n_samples, 1) containing the labels.
        test_size (float, optional): proportion of the data to be used for testing. Defaults to 0.2.
        shuffle (bool, optional): whether to shuffle the data before splitting. Defaults to True.
        seed (int, optional): seed for the random number generator. Defaults to None.

    Returns:
        tuple: containing the train and test sets.
    """
    if shuffle:
        np.random.seed(seed)
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
    split_index = int(len(X) * (1 - test_size))
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

def r2_score(y_true, y_pred):
    """R2 score.

    Args:
        y_true (np.array): of shape (n_samples, 1) containing the true labels.
        y_pred (np.array): of shape (n_samples, 1) containing the predictions.

    Returns:
        float: the R2 score.
    """
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def accuracy_score(y_true, y_pred):
    """Accuracy score.

    Args:
        y_true (np.array): of shape (n_samples, 1) containing the true labels.
        y_pred (np.array): of shape (n_samples, 1) containing the predictions.

    Returns:
        float: the accuracy score.
    """
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    """Confusion matrix.

    Args:
        y_true (np.array): of shape (n_samples, 1) containing the true labels.
        y_pred (np.array): of shape (n_samples, 1) containing the predictions.

    Returns:
        np.array: of shape (2, 2) containing the confusion matrix.
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    return np.array([[true_positives, false_positives], [false_negatives, true_negatives]])

def plot_confusion_matrix(y_true, y_pred, figsize=(5, 5), cmap=plt.cm.Blues):
    """Plot the confusion matrix.

    Args:
        y_true (np.array): of shape (n_samples, 1) containing the true labels.
        y_pred (np.array): of shape (n_samples, 1) containing the predictions.
        cmap (matplotlib colormap, optional): colormap. Defaults to plt.cm.Blues.
        n_classes (int, optional): number of classes. Defaults to 2.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.style.use('default')
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Positive(1)', 'Negative(0)'])
    plt.yticks([0, 1], ['Positive(1)', 'Negative(0)'])
    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.show()


def cluster_plot(X, clusterid, centroids=None, y=None):
    X = np.asarray(X)
    cls = np.asarray(clusterid)
    if y is None:
        y = np.zeros((X.shape[0], 1))
    else:
        y = np.asarray(y)
    if centroids is not None:
        centroids = np.asarray(centroids)
    K = np.size(np.unique(cls))
    C = np.size(np.unique(y))
    ncolors = np.max([C, K])

    # plot data points color-coded by class, cluster markers and centroids
    colors = [0]*ncolors
    for color in range(ncolors):
        colors[color] = plt.cm.jet.__call__((color*255)//(ncolors-1))[:3]
    for i, cs in enumerate(np.unique(y)):
        plt.plot(X[(y == cs).ravel(), 0], X[(y == cs).ravel(), 1], 'o',
                 markeredgecolor='k', markerfacecolor=colors[i], markersize=4,
                 zorder=2)
    for i, cr in enumerate(np.unique(cls)):
        plt.plot(X[(cls == cr).ravel(), 0], X[(cls == cr).ravel(), 1], 'o',
                 markersize=12, markeredgecolor=colors[i],
                 markerfacecolor='None', markeredgewidth=3, zorder=1)
    if centroids is not None:
        for cd in range(centroids.shape[0]):
            plt.plot(centroids[cd, 0], centroids[cd, 1], '*', markersize=22,
                     markeredgecolor='k', markerfacecolor=colors[cd],
                     markeredgewidth=2, zorder=3)

    # create legend
    legend_items = (np.unique(y).tolist() + np.unique(cls).tolist() +
                    np.unique(cls).tolist())
    for i in range(len(legend_items)):
        if i < C:
            legend_items[i] = 'Class: {0}'.format(legend_items[i])
        elif i < C + K:
            legend_items[i] = 'Cluster: {0}'.format(legend_items[i])
        else:
            legend_items[i] = 'Centroid: {0}'.format(legend_items[i])
    plt.legend(legend_items, numpoints=1, markerscale=.75, prop={
               'size': 9}, loc='center left', bbox_to_anchor=(1, 0.5))
