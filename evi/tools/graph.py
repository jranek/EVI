import igraph as ig
import leidenalg
import numpy as np
import scipy

def get_igraph(W = None,
               directed: bool = None):
    """Converts adjacency matrix into igraph object

    Parameters
    W: (default = None)
        adjacency matrix
    directed: bool (default = None)
        whether graph is directed or not
    ----------

    Returns
    g: ig.Graph
        graph of adjacency matrix
    ----------
    """
    sources, targets = W.nonzero()
    weights = W[sources, targets]
    if type(weights) == np.matrix:
        weights = weights.A1 #flattens 
    g = ig.Graph(directed = directed)
    g.add_vertices(np.shape(W)[0])
    g.add_edges(list(zip(sources, targets)))
    g.es['weight'] = weights  
    
    return g

def cluster_leiden(g: ig.Graph,
                   resolution: float = 0.5,
                   random_state: int = 0):
    """performs leiden clustering using RBConfigurationVertexPartition

    Parameters
    g: ig.Graph
        igraph object
    resolution: float (default = 0.5)
        resolution parameter for clustering
    random_state: int (default = 0)
        integer for reproducibility
    ----------

    Returns
    membership: list
        cluster membership list
    ----------
    """
    partition_type = leidenalg.RBConfigurationVertexPartition
    partition = leidenalg.find_partition(g, partition_type, resolution_parameter = resolution, n_iterations = -1,
                                         seed = random_state, weights =  np.array(g.es['weight']).astype(np.float64))
    membership = partition.membership
    
    return membership

def compute_laplacian(W = None,
                      normalized: bool = False):
    """computes normalized or unnormalized graph laplacian

    Parameters
    W: Matrix (default = None)
        adjacency matrix
    normalized: bool (default = None)
        whether to compute normalized or unnormalized graph laplacian
    ----------

    Returns
    L: Matrix
        graph laplacian matrix
    ----------
    """
    g = get_igraph(W, directed = False)
    L = g.laplacian(weights =  np.array(g.es['weight']).astype(np.float64), normalized = normalized)
    L = scipy.sparse.csr_matrix(np.reshape(L, np.shape(W)))
    return L