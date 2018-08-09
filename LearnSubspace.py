import numpy as np


def learn_subspace_by_pca(X, k):
    X_zeromean, X_mean = normalize_data(X)
    V = compute_pca(X_zeromean, k)
    return X_zeromean, X_mean, V


def normalize_data(X):
    X_mean = np.mean(X,axis=1)[:,np.newaxis]  # normalize X
    # X_mean = np.outer(X_mean,np.ones(X.shape[1]))
    # X_norm = X - X_mean
    #X_norm = X  # remove this line in order to normalize X
    X_zeromean = X-X_mean
    return X_zeromean, X_mean


def compute_pca(X,k):
    V, S, U = np.linalg.svd(X)  # compute PCA
    V = V[:, :k]
    return V