import numpy as np
import cv2
import copy
from Plots import open_figure, PlotImages


def main():
    d, k, N = update_params()

    """Read one frame"""
    X, height, width = prepare_image('background.jpeg')
    d = height * width

    """Generate outliers (moving object)"""
    X_ol = X.copy()
    start_idx = np.array([20,20])
    end_idx = np.array([90,90])
    W = np.eye(d)
    for i in range(start_idx[0], end_idx[0]):
        for j in range(start_idx[1], end_idx[1]):
            X_ol[i][j] = -1
            idx = np.ravel_multi_index((i, j), X_ol.shape)
            W[idx, idx] = 0

    X = np.reshape(X, (d, 1))
    X_ol = np.reshape(X_ol,(d, 1))

    """Compute PCA (SVD on X)"""
    V, S, U = np.linalg.svd(X)
    V = V[:, :k]

    """Solve WLS (w=1 for background pixels, w=0 of object pixels)"""
    #scipy.linalg.lstsq OR scipy.sparse.linalg.lsqr
    WV = W @ V
    WX = W @ X_ol
    alpha = np.linalg.lstsq(WV,WX, rcond=None)[0]
    #print('alpha: ', alpha)

    """Projection on the subspace"""
    proj = V @ V.T @ X
    error = X - proj
    proj_ol = V @ V.T @ X_ol
    error_ol = X_ol - proj_ol
    proj_wls = V @ alpha
    error_wls = X_ol - proj_wls

    """Plot one frame"""
    plt_image = np.reshape(X, (height, width))
    plt_proj = np.reshape(proj, (height, width))
    plt_error = np.reshape(error, (height, width))

    plt_image_ol = np.reshape(X_ol, (height, width))
    plt_proj_ol = np.reshape(proj_ol, (height, width))
    plt_error_ol = np.reshape(error_ol, (height, width))

    plt_image_ = np.reshape(X_ol, (height, width))
    plt_proj_wls = np.reshape(proj_wls, (height, width))
    plt_error_wls = np.reshape(error_wls, (height, width))

    open_figure(1,'Original Image',(7,7))
    PlotImages(1,3,3,1,[plt_image, plt_image_ol, plt_image_ol, plt_proj, plt_proj_ol, plt_proj_wls, plt_error, plt_error_ol, plt_error_wls],
               ['Original Image', 'Image with outliers', 'Image with outliers', 'Projection', 'Projection', 'Projection', 'Error', 'Error', 'Error'],
               'gray',axis=True,colorbar=True, m=300)


def update_params():
    d = 100
    k = 5
    N = 500
    return d, k, N


def sample_points_in_subspace(V, k, N):
    mu = np.zeros(k)
    alpha = np.random.multivariate_normal(mu,np.eye(k),N).T
    X = V @ alpha
    return X


def generate_iid_noise(d, N, sigma):
    mu = np.zeros(d)
    n = np.random.multivariate_normal(mu,np.eye(d) * (sigma ** 2),N).T
    return n


def prepare_image(filepath):
    I = cv2.imread(filepath,0)
    I = cv2.resize(I, (0,0), fx=0.5, fy=0.5)
    I = I.astype(np.float32)
    print('Image shape: ', I.shape)
    height, width = I.shape
    return I, height, width


if __name__ == "__main__":
    main()