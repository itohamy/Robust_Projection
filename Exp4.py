import numpy as np
import cv2
import maxflow
from Plots import open_figure, PlotImages
import matplotlib.pyplot as plt
from video_to_frames import extractImages


np.random.seed(17)


def main():
    d, k, N, data_path = update_params()

    """ Build X in shape dxN: Read N frames from video """
    #extractImages("spity.mp4", data_path)
    X_, m, n = prepare_image(data_path + 'frame0.jpg')
    d = m * n
    print("d:", d, "  k:", k, "  N:", N, "  Image shape:", X_.shape)
    X = np.reshape(X_, (d, 1))
    for i in range(1, N):
        v_, _, _ = prepare_image(data_path + 'frame%d.jpg' % i)
        v = np.reshape(v_, (d, 1))
        X = np.column_stack((X, v))

    """Compute PCA (SVD on X)"""
    X_mean = np.mean(X, axis=1)  # normalize X
    X_mean = np.outer(X_mean, np.ones(N))
    X_norm = X - X_mean
    X_norm = X  # remove this line in order to normalize X
    V, S, U = np.linalg.svd(X_norm)  # compute PCA
    V = V[:, :k]

    """ Generate outliers (moving object) for one frame and keep the ground truth W """
    X_o = X_norm[:,9:10]
    X_o = np.reshape(X_o, (m, n))
    start_idx = np.array([20, 20])
    end_idx = np.array([70, 70])
    W = np.eye(d)
    for i in range(start_idx[0], end_idx[0]):
        for j in range(start_idx[1], end_idx[1]):
            X_o[i][j] = -10 + (5)*np.random.standard_normal()

            idx = np.ravel_multi_index((i, j), X_o.shape)
            W[idx, idx] = 0
    X_o = np.reshape(X_o, (d, 1))

    """ Solve WLS (w=1 for background pixels, w=0 of object pixels) - this is the ground truth alpha"""
    WV = W @ V
    WX_o = W @ X_o
    alpha_o = np.linalg.lstsq(WV, WX_o, rcond=None)[0]
    # print(alpha_o)
    # alpha_o += (100000)*np.random.standard_normal(alpha_o.size).reshape(k,1)
    # print(alpha_o)
    print("X_norm shape:", X_norm.shape, "  V shape:", V.shape, "  alpha_o shape:", alpha_o.shape, "  W shape:", W.shape)

    """ Projection on the subspace """
    proj = V @ V.T @ X_norm[:, :1]
    error = X_norm[:, :1] - proj
    proj_o = V @ V.T @ X_o
    error_o = X_o - proj_o
    proj_robust = V @ alpha_o
    error_robust = X_o - proj_robust

    """Plot one frame"""
    plt_image = np.reshape(X_norm[:, :1], (m, n))
    plt_proj = np.reshape(proj, (m, n))
    plt_error = np.reshape(error, (m, n))

    plt_image_o = np.reshape(X_o, (m, n))
    plt_proj_o = np.reshape(proj_o, (m, n))
    plt_error_o = np.reshape(error_o, (m, n))

    plt_image_robust = np.reshape(X_o, (m, n))
    plt_proj_robust = np.reshape(proj_robust, (m, n))
    plt_error_robust = np.reshape(error_robust, (m, n))

    """ Compute weights by Graph-Cut """
    # Create the graph
    g = maxflow.Graph[int](m, n)
    nodes = g.add_nodes(m * n)
    sigma_BG = 0.4
    sigma_FG = 1
    # go through all nodes and add edges
    for i in range(m * n):
        # add edge source->pixels and pixels->sink:
        source_weight = ((1./(sigma_BG)**2)*(abs(X_o[i]-proj_robust[i])))   # X_o[i]  # BG probability
        sink_weight = (X_o[i]**2)/(sigma_FG**2)   # 255 - X_o[i]  # FG probability
        g.add_tedge(i, source_weight, sink_weight)
        if i == 3859 or i == 3860 or i == 3861 or i == 1970 or i == 7830 or i == 19300:
            print(i, source_weight, sink_weight, X_o[i], proj_robust[i], X_o[i]-proj_robust[i])
        # add edges between pixels neighbors
        if i % n != 0:  # left exists
            edge_wt = 2
            g.add_edge(i, i - 1, edge_wt, edge_wt)
        if (i + 1) % n != 0:  # right exists
            edge_wt = 2
            g.add_edge(i, i + 1, edge_wt, edge_wt)
        if i // n != 0:  # up exists
            edge_wt = 2
            g.add_edge(i, i - n, edge_wt, edge_wt)
        if i // n != m - 1:  # down exists
            edge_wt = 2
            g.add_edge(i, i + n, edge_wt, edge_wt)
    # compute min-cut / max-a-posterior
    flow = g.maxflow()
    sgm = g.get_grid_segments(nodes)  # Get the segments.
    sgm_ = np.reshape(sgm,(m,n))
    print(sgm_)
    print("flow: ",flow)

    print('\n3859',sgm[3859])
    print('3860',sgm[3860])
    print('3861',sgm[3861])
    print('1970',sgm[1970])
    print('7830',sgm[7830])
    print('19300',sgm[19300])

    # converting the True/False to Pixel intensity
    out = np.ones((m,n))
    for i in range(m):
        for j in range(n):
            if sgm_[i,j]:
                out[i,j] = -1 # background
            else:
                out[i,j] = 1 # foreground

    """ Display results """
    open_figure(1,'Projections', (7,7))
    PlotImages(1,3,3,1,[plt_image, plt_image_o, plt_image_robust, plt_proj, plt_proj_o, plt_proj_robust, plt_error, plt_error_o, plt_error_robust],
               ['Direct Proj., X', 'Direct Proj., X_o', 'Robust Proj., X_o', 'Proj.', 'Proj.', 'Proj.', 'Error', 'Error', 'Error'],
               'gray',axis=True, colorbar=True, m=100)
    open_figure(2,'Graph Cut', (5,5))
    PlotImages(2,1,1,1,[out],
               ['Graph Cut'],
               'gray',axis=True,colorbar=True)
    plt.show()


def update_params():
    d = 100
    k = 5
    N = 20
    data_path = "frames/"
    return d, k, N, data_path


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
    I = cv2.resize(I, (0,0), fx=0.1, fy=0.1)
    I = I.astype(np.float32)
    #print('Image shape: ', I.shape)
    height, width = I.shape
    return I, height, width


if __name__ == "__main__":
    main()