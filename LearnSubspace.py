import numpy as np


def normalize_data(X):
    X_mean = np.mean(X,axis=1)  # normalize X
    X_mean = np.outer(X_mean,np.ones(X.shape[1]))
    X_norm = X - X_mean
    X_norm = X  # remove this line in order to normalize X
    return X_norm


def compute_pca(X,k):
    V, S, U = np.linalg.svd(X)  # compute PCA
    V = V[:, :k]
    return V