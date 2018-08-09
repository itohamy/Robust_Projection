import numpy as np
import maxflow
from Bunch import Bunch

def compute_graphcut(X, proj, m, n):
    # Create the graph
    g = maxflow.Graph[int](m, n)
    nodes = g.add_nodes(m * n)
    sigma_BG = 5
    sigma_FG = 10
    lambda_ = 2
    source_weight = np.zeros_like(X)
    sink_weight = np.zeros_like(X)
    # go through all nodes and add edges
    #sigma_BG=sigma_FG=5 # DEBUG
    for i in range(m * n):
        # add edge source->pixels and pixels->sink:
        source_weight[i] = ((1./(sigma_BG)**2)*((X[i]-proj[i])**2))   # X_o[i]  # BG probability
        sink_weight[i] = (X[i]**2)/(sigma_FG**2)   # 255 - X_o[i]  # FG probability
        g.add_tedge(i, source_weight[i], sink_weight[i])
        if i == 3859 or i == 3860 or i == 3861 or i == 1970 or i == 7830 or i == 19300:
            print(i, source_weight[i], sink_weight[i], X[i], proj[i], X[i]-proj[i])
        # add edges between pixels neighbors
        if i % n != 0:  # left exists
            edge_wt = lambda_
            g.add_edge(i, i - 1, edge_wt, edge_wt)
        if (i + 1) % n != 0:  # right exists
            edge_wt = lambda_
            g.add_edge(i, i + 1, edge_wt, edge_wt)
        if i // n != 0:  # up exists
            edge_wt = lambda_
            g.add_edge(i, i - n, edge_wt, edge_wt)
        if i // n != m - 1:  # down exists
            edge_wt = lambda_
            g.add_edge(i, i + n, edge_wt, edge_wt)

    # compute min-cut / max-a-posterior
    flow = g.maxflow()

    # Get the segmentation
    sgm = g.get_grid_segments(nodes)
    sgm_ = np.reshape(sgm, (m, n))
    print(sgm_)
    print("flow: ", flow)

    print('\n3859', sgm[3859])
    print('3860', sgm[3860])
    print('3861', sgm[3861])
    print('1970', sgm[1970])
    print('7830', sgm[7830])
    print('19300', sgm[19300])

    # converting the True/False to Pixel intensity
    out = np.ones((m,n))
    for i in range(m):
        for j in range(n):
            if sgm_[i,j]:
                out[i,j] = -1 # background
            else:
                out[i,j] = 1 # foreground
    b = Bunch()
    b.out = out
    b.source_weight = source_weight
    b.sink_weight = sink_weight
    return b
