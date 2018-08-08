import numpy as np
import cv2
import maxflow
from Plots import open_figure, PlotImages
import matplotlib.pyplot as plt

np.random.seed(17)

def main():
    d, k, N = update_params()

    """Read one frame"""
    X, m, n = prepare_image('background.jpeg')
    d = m * n

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
    print(V.size)

    """Solve WLS (w=1 for background pixels, w=0 of object pixels)"""
    WV = W @ V
    WX = W @ X_ol
    alpha = np.linalg.lstsq(WV,WX, rcond=None)[0]
    print(alpha)
    alpha += (4164) * np.random.standard_normal(alpha.size)
    print(alpha)

    """Projection on the subspace"""
    proj = V @ V.T @ X
    error = X - proj
    proj_ol = V @ V.T @ X_ol
    error_ol = X_ol - proj_ol
    proj_wls = V @ alpha
    error_wls = X_ol - proj_wls

    """Plot one frame"""
    plt_image = np.reshape(X, (m, n))
    plt_proj = np.reshape(proj, (m, n))
    plt_error = np.reshape(error, (m, n))

    plt_image_ol = np.reshape(X_ol, (m, n))
    plt_proj_ol = np.reshape(proj_ol, (m, n))
    plt_error_ol = np.reshape(error_ol, (m, n))

    plt_image_ = np.reshape(X_ol, (m, n))
    plt_proj_wls = np.reshape(proj_wls, (m, n))
    plt_error_wls = np.reshape(error_wls, (m, n))


    """ Compute weights by Graph-Cut """
    # Create the graph
    g = maxflow.Graph[int](m, n)
    nodes = g.add_nodes(m * n)

    sigma1 = 0.4
    sigma2 = 6
    # go through all nodes and add edges
    for i in range(m * n):
        # add edge source->pixels and pixels->sink:
        source_weight = ((1./(sigma1)**2)*(X_ol[i]-proj_wls[i])**2)   # X_ol[i]  # BG probability
        sink_weight = (X_ol[i]**2)/(sigma2**2)   # 255 - X_ol[i]  # FG probability
        g.add_tedge(i, source_weight, sink_weight)
        if(i == 0 or i == 6900):
            print("sink/src links: ", i, source_weight, sink_weight, X_ol[i]-proj_wls[i])

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
    sgm_ = np.reshape(sgm, (m, n))
    print(sgm_)
    print("flow: ", flow)

    print('\n0',sgm[0])
    print('6900',sgm[6900])

    out = np.ones((m,n))
    for i in range(m):
        for j in range(n): # converting the True/False to Pixel intensity
            if sgm_[i,j]:
                out[i,j] = -1 # background
            else:
                out[i,j] = 1 # foreground

    open_figure(1,'Original Image', (7,7))
    PlotImages(1,4,3,1,[plt_image, plt_image_ol, plt_image_ol, plt_proj, plt_proj_ol, plt_proj_wls, plt_error, plt_error_ol, plt_error_wls],
               ['Original Image', 'Image with outliers', 'Image with outliers', 'Projection', 'Projection', 'Projection', 'Error', 'Error', 'Error'],
               'gray',axis=True,colorbar=True, m=300)

    open_figure(2,'Graph Cut', (5,5))
    PlotImages(2,1,1,1,[out],
               ['Graph Cut'],
               'gray',axis=True,colorbar=True)

    plt.show()


def update_params():
    d = 100
    k = 1
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