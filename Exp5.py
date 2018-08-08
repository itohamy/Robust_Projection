import numpy as np
import cv2
from Plots import open_figure, PlotImages
import matplotlib.pyplot as plt
from LearnSubspace import learn_subspace_by_pca
from video_to_frames import extractImages
from GraphCut import compute_graphcut

np.random.seed(17)


def main():
    k, N, data_path = update_params()

    """ Build X in shape dxN: Read N frames from video """
    X, d, m, n = build_data_matrix(data_path, k, N)

    """Compute PCA (SVD on X)"""
    X_norm, V = learn_subspace_by_pca(X, k)

    """ Generate outliers (moving object) for one frame and keep the ground truth W """
    X_o, W = generate_outliers(X_norm, d, m, n)

    """ Solve WLS (w=1 for background pixels, w=0 of object pixels) - this is the ground truth alpha"""
    alpha_o = solve_wls(W, V, X_o)
    # print(alpha_o)
    # alpha_o += (100000)*np.random.standard_normal(alpha_o.size).reshape(k,1)
    # print(alpha_o)
    print("X_norm shape:", X_norm.shape, "  V shape:", V.shape, "  alpha_o shape:", alpha_o.shape, "  W shape:", W.shape)

    """ Projection on the subspace """
    proj, error = project_on_subspace(X_norm[:, :1], V, V.T @ X_norm[:, :1])
    proj_o, error_o = project_on_subspace(X_o, V, V.T @ X_o)
    proj_robust, error_robust = project_on_subspace(X_o, V, alpha_o)

    """ Plot projections and errors """
    plt_image, plt_proj, plt_error = prepare_projection_to_display(X_norm[:, :1], proj, error, m, n)
    plt_image_o, plt_proj_o, plt_error_o = prepare_projection_to_display(X_o, proj_o, error_o, m, n)
    plt_image_robust, plt_proj_robust, plt_error_robust = prepare_projection_to_display(X_o, proj_robust, error_robust, m, n)

    """ Compute Graph Cut to get W """
    gc_out = compute_graphcut(X_o, proj_robust, m, n)

    """ Display results """
    open_figure(1,'Projections', (7,7))
    PlotImages(1,3,3,1,[plt_image, plt_image_o, plt_image_robust, plt_proj, plt_proj_o, plt_proj_robust, plt_error, plt_error_o, plt_error_robust],
               ['Direct Proj., X', 'Direct Proj., X_o', 'Robust Proj., X_o', 'Proj.', 'Proj.', 'Proj.', 'Error', 'Error', 'Error'],
               'gray',axis=True, colorbar=True, m=200)
    open_figure(2,'Graph Cut', (5,5))
    PlotImages(2,1,1,1,[gc_out],
               ['Graph Cut'],
               'gray',axis=True,colorbar=True)
    plt.show()


def update_params():
    k = 5
    N = 20
    data_path = "frames/"
    return k, N, data_path


def prepare_image(filepath):
    I = cv2.imread(filepath,0)
    I = cv2.resize(I, (0,0), fx=0.1, fy=0.1)
    I = I.astype(np.float32)
    height, width = I.shape
    return I, height, width


def generate_outliers(X_norm, d, m, n):
    X_o = X_norm[:,9:10]
    X_o = np.reshape(X_o, (m, n))
    start_idx = np.array([20, 20])
    end_idx = np.array([70, 70])
    W = np.eye(d)
    for i in range(start_idx[0],end_idx[0]):
        for j in range(start_idx[1],end_idx[1]):
            X_o[i][j] = -10 + 5 * np.random.standard_normal()
            idx = np.ravel_multi_index((i, j), X_o.shape)
            W[idx, idx] = 0
    X_o = np.reshape(X_o, (d, 1))
    return X_o, W


def solve_wls(W, V, X):
    WV = W @ V
    WX = W @ X
    alpha = np.linalg.lstsq(WV,WX,rcond=None)[0]
    return alpha


def build_data_matrix(data_path, k, N):
    #extractImages("spity.mp4", data_path)
    X_, m, n = prepare_image(data_path + 'frame0.jpg')
    d = m * n
    print("d:",d,"  k:",k,"  N:",N,"  Image shape:",X_.shape)
    X = np.reshape(X_,(d,1))
    for i in range(1,N):
        v_, _, _ = prepare_image(data_path + 'frame%d.jpg' % i)
        v = np.reshape(v_,(d,1))
        X = np.column_stack((X,v))
    return X, d, m, n


def project_on_subspace(X, V, alpha):
    proj = V @ alpha
    error = X - proj
    return proj, error


def prepare_projection_to_display(X, proj, error, m, n):
    plt_image = np.reshape(X, (m, n))
    plt_proj = np.reshape(proj, (m, n))
    plt_error = np.reshape(error, (m, n))
    return plt_image, plt_proj, plt_error

if __name__ == "__main__":
    main()