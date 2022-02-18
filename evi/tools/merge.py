import sys
import evi
from anndata import AnnData
import scvelo as scv
import scanpy as sc
import pandas as pd
import numpy as np
import phate
import graphtools
from precise import PVComputation, IntermediateFactors, ConsensusRepresentation
import sklearn
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import scipy
from scipy.stats import ks_2samp
from scipy.linalg import *

def expression(adata: AnnData,
                x1key: str = 'X',
                X1 = None,
                logX1: bool = False,
                k: int = None,
                n_pcs: int = None,
                **args):
    """Obtain graph and embedding from expression data

    Parameters
    adata: AnnData
        Annotated data object
    x1key: str (default = 'X')
        string referring to the layer of first matrix. Can be X, spliced or None
    X1: (default = None)
        matrix referring to the first data type
    logX1: bool (default = False)
        boolean referring to whether the first data type should be log transformed
    k: int (default = None)
        integer referring to the number of nearest neighbors to use if graph not provided
    n_pcs: int (default = None)
        integer referring to the number of principal components
            if == 0, use all data for nearest neighbors
    ----------

    Returns
    W: ndarray
        sparse symmetric adjacency matrix of expression data. Dimensions cells x cells
    embedding: 
        PCA embedding. Dimensions cells x n_pcs 
    ----------
    """
    if X1 is None:
        if x1key == 'X':
            X1 = adata.X.copy()
        elif x1key == 'spliced':
            X1 = adata.layers['spliced'].copy()
        else:
            sys.stdout.write('X1 is not specified')
  
    if scipy.sparse.issparse(X1) == False:
        X1 = scipy.sparse.csr_matrix(X1)
    if logX1 == True:
        X1 = scipy.sparse.csr_matrix.log1p(X1)

    adata_pp = AnnData(X1)
    sc.tl.pca(adata_pp, n_comps = n_pcs, zero_center = True, svd_solver = 'arpack', random_state = 0) #random state set
    scv.pp.neighbors(adata_pp, n_neighbors=k, n_pcs=n_pcs)
    W = check_symmetric(W = adata_pp.obsp['connectivities'])
    embedding = np.asarray(adata_pp.obsm['X_pca'])

    return W, embedding

def moments(adata: AnnData,
                x1key: str = 'Ms',
                X1 = None,
                k: int = None,
                n_pcs: int = None,
                **args):
    """Obtain graph and embedding from spliced moments

    Parameters
    adata: AnnData
        Annotated data object
    x1key: str (default = 'Ms')
        string referring to the layer of first matrix. Should be Ms or moments equivalent
    X1: (default = None)
        matrix referring to the first data type 
    k: int (default = None)
        integer referring to the number of nearest neighbors to use if graph not provided
    n_pcs: int (default = None)
        integer referring to the number of principal components
            if == 0, use all data for nearest neighbors
    ----------

    Returns
    W: ndarray
        sparse symmetric adjacency matrix of expression data. Dimensions cells x cells
    embedding: 
        PCA embedding. Dimensions cells x n_pcs 
    ----------
    """
    X1 = scipy.sparse.csr_matrix(adata.layers[x1key]) 
    adata_ =  AnnData(X1)
    adata_.obs_names = adata.obs_names.copy()
    adata_.var_names = adata.var_names.copy()
    sc.pp.pca(adata_, n_comps = n_pcs, zero_center = True, svd_solver = 'arpack', random_state = 0)
    scv.pp.neighbors(adata_, n_neighbors = k, n_pcs = n_pcs, knn = True, random_state = 0)
    W = check_symmetric(W = adata_.obsp['connectivities'])
    embedding = np.asarray(adata_.obsm['X_pca'])
    return W, embedding

def concat_merge(adata: AnnData,
                x1key: str = 'Ms',
                x2key: str = 'velocity',
                X1 = None,
                X2 = None,
                logX1: bool = False,
                logX2: bool = False,
                k: int = 10,
                n_pcs: int = 50,
                **args):

    """merges data types through horizontal concatenation.

    Parameters
    x1key: str (default = 'Ms')
        string referring to the layer of first matrix. Can be X, Ms, spliced, unspliced, velocity, or None
    x2key: str (default = 'velocity')
        string referring to the layer of second matrix. Can be X, Ms, spliced, unspliced, velocity, or None
    X1: (default = None)
        matrix referring to the first data type
    X2: (default = None)
        matrix referring to the second data type
    logX1: bool (default = False)
        boolean referring to whether the first data type should be log transformed
    logX2: bool (default = False)
        boolean referring to whether the second data type should be log transformed
    k: int (default = None)
        integer referring to the number of nearest neighbors to use if graph not provided
    n_pcs: int (default = None)
        integer referring to the number of principal components
            if == 0, use all data for nearest neighbors
    ----------

    Returns
    W: ndarray
        sparse symmetric adjacency matrix from horizontally concatenated data. Dimensions cells x cells
    embedding: 
        PCA embedding. Dimensions cells x n_pcs 
    ----------
    """    
    X1, X2 = check_inputs(adata, x1key = x1key, x2key = x2key, X1 = X1, X2 = X2)
    X1, X2 = check_log(X1 = X1, X2 = X2, logX1 = logX1, logX2 = logX2)
    X1, X2 = check_sparse(X1 = X1, X2 = X2, return_sparse = True)
        
    X_merge = scipy.sparse.hstack((X1, X2)) #n x 2*p

    adata_merge =  AnnData(X_merge.todense()) #needs to be dense or throws an error 
    adata_merge.obs_names = adata.obs_names.copy()
    adata_merge.var_names = np.concatenate([adata.var_names, adata.var_names])
    sc.pp.pca(adata_merge, n_comps = n_pcs, zero_center = True, svd_solver = 'arpack', random_state = 0)
    scv.pp.neighbors(adata_merge, n_neighbors = k, n_pcs = n_pcs, knn = True, random_state = 0)
    W = check_symmetric(W = adata_merge.obsp['connectivities']) #will convert to sparse 
    embedding = np.asarray(adata_merge.obsm['X_pca'])

    return W, embedding

def sum_merge(adata: AnnData,
                x1key: str = 'Ms',
                x2key: str = 'velocity',
                X1 = None,
                X2 = None,
                logX1: bool = False,
                logX2: bool = False,
                k: int = 10,
                n_pcs: int = 50,
                **args):
    """merges data types through summing matrices.

    Parameters
    x1key: str (default = 'Ms')
        string referring to the layer of first matrix. Can be X, Ms, spliced, unspliced, velocity, or None
    x2key: str (default = 'velocity')
        string referring to the layer of second matrix. Can be X, Ms, spliced, unspliced, velocity, or None
    X1: (default = None)
        matrix referring to the first data type
    X2: (default = None)
        matrix referring to the second data type
    logX1: bool (default = False)
        boolean referring to whether the first data type should be log transformed
    logX2: bool (default = False)
        boolean referring to whether the second data type should be log transformed
    k: int (default = None)
        integer referring to the number of nearest neighbors to use if graph not provided
    n_pcs: int (default = None)
        integer referring to the number of principal components
            if == 0, use all data for nearest neighbors
    ----------

    Returns
    W: ndarray
        sparse symmetric adjacency matrix from summed data. Dimensions cells x cells
    embedding: 
        PCA embedding. Dimensions cells x n_pcs 
    ----------
    """  
    X1, X2 = check_inputs(adata, x1key = x1key, x2key = x2key, X1 = X1, X2 = X2)
    X1, X2 = check_log(X1 = X1, X2 = X2, logX1 = logX1, logX2 = logX2)
    X1, X2 = check_sparse(X1 = X1, X2 = X2, return_sparse = True)

    X_merge = X1 + X2 #n x p

    adata_merge =  AnnData(X_merge) #needs to be dense or throws an error 
    adata_merge.obs_names = adata.obs_names.copy()
    adata_merge.var_names = adata.var_names.copy()
    sc.pp.pca(adata_merge, n_comps = n_pcs, zero_center = True, svd_solver = 'arpack', random_state = 0)
    scv.pp.neighbors(adata_merge, n_neighbors = k, n_pcs = n_pcs, knn = True, random_state = 0)
    W = check_symmetric(W = adata_merge.obsp['connectivities'])
    embedding = np.asarray(adata_merge.obsm['X_pca'])
    return W, embedding

def cellrank(adata: AnnData,
            x1key: str = 'Ms',
            x2key: str = 'velocity',
            lam: float = 0.2,
            scheme: str = 'correlation',
            mode: str = 'deterministic',
            k: int = None,
            n_pcs: int = None,
            **args):
    """computes a weighted transition probability matrix through cellrank algorithm: https://cellrank.readthedocs.io/en/stable/

    Parameters
    adata: AnnData
        Annotated data object
    x1key: str (default = 'Ms')
        string referring to the layer of first matrix. Can be X, Ms, spliced
    x2key: str (default = 'velocity')
        string referring to the layer of second matrix. Must be velocity
    lam: float (default = 0.2)
        float referring to the weight given to the connectivities matrix in weighted sum
    scheme: str (default = 'correlation')
        string referring to the similarity between velocity vectors and expression displacement. Can be correlation, dot_product, or cosine
    mode: str (default = 'deterministic')
        string referring to the mode of velocity transition matrix computation. Can be deterministic, stochastic, monte_carlo, or sampling.
            all modes apart from deterministic take into account velocity uncertainty
    ----------

    Returns
    W: ndarray
        sparse symmetric transition probability matrix of weighted sum between expression transition matrix and velocity transition matrix. Dimensions cells x cells
    embedding: None
    ----------
    """  
    from cellrank.tl.kernels import VelocityKernel, ConnectivityKernel

    if x2key != 'velocity':
        sys.stderr.write('x2key must be velocity. respecify')
        pass

    vk = VelocityKernel(adata, xkey = x1key, vkey = x2key).compute_transition_matrix(scheme = scheme, mode = mode) #n x n 
    ck = ConnectivityKernel(adata).compute_transition_matrix() #n x n 
    
    combined_kernel = (((1-lam) * ck) + (lam * vk)).compute_transition_matrix()
    combined_kernel.write_to_adata(key="combined_kernel")
    combined_kernel = adata.obsp["combined_kernel"] 
    W = combined_kernel.copy()
    W = check_symmetric(W)
    
    return W, None

def snf(adata: AnnData,
        x1key: str = 'Ms',
        x2key: str = 'velocity',
        metric: str = 'euclidean',
        k: int = 10,
        k_embed: int = 10,
        mu: float = 0.5,
        X1 = None,
        X2 = None,
        logX1: bool = False,
        logX2: bool = False,
        K: int = None,
        **args):

    """merges data types through similarity network fusion: https://www.nature.com/articles/nmeth.2810
       implemented in python: https://snfpy.readthedocs.io/en/latest/#
       Following snf, eigendecomposition is performed on the Laplacian to obtain a k nearest neighbor graph 

    Parameters
    adata: AnnData
        Annotated data object
    x1key: str (default = 'Ms')
        string referring to the layer of first matrix. Can be X, Ms, spliced, unspliced, velocity, or None
    x2key: str (default = 'velocity')
        string referring to the layer of second matrix. Can be X, Ms, spliced, unspliced, velocity, or None
    metric: str (default = 'euclidean')
        string referring to the distance computation in affinity graph construction   
    k: int (default = 10)
        integer referring to the number of nearest neighbors in k-nearest neighbor graph in heat kernel
    k_embed: int (default = 10)
        integer referring to the number of nearest neighbors in k-nearest neighbor graph in embedding
    mu: float (deafult = 0.05)
        float referring to bandwidth scale in heat kernel
    X1: (default = None)
        matrix referring to the first data type
    X2: (default = None)
        matrix referring to the second data type
    logX1: bool (default = False)
        boolean referring to whether the first data type should be log transformed
    logX2: bool (default = False)
        boolean referring to whether the second data type should be log transformed
    K: int (default = None)
        number of eigenvectors to compute for graph embedding
    return_adj: bool (default = False)
        whether to return adjacency matrix. If False, a kNN graph is constructed from the embedding
    ----------

    Returns
    W: ndarray
        joint embedding knn graph
    embedding: 
        Laplacian embedding of joint data
    ----------
    """  
    from snf import compute

    X1, X2 = check_inputs(adata, x1key = x1key, x2key = x2key, X1 = X1, X2 = X2)
    X1, X2 = check_log(X1 = X1, X2 = X2, logX1 = logX1, logX2 = logX2)
    X1, X2 = check_sparse(X1 = X1, X2 = X2, return_sparse = False) #requires dense :(
              
    C = [X1, X2]
    adj_i = compute.make_affinity(C, metric = metric, K = k, mu = mu) #2*n x n
    adj_j = compute.snf(adj_i, K = k) #n x n

    adj_j = check_symmetric(adj_j) #make sparse and symmetric
    L = evi.tl.compute_laplacian(adj_j, normalized = False) #non-normalized laplacian
    _, eigenvecs = eigendecomposition(W = L, K = K, which = 'SM') #first k eigenvecs of laplacian

    if eigenvecs is None: #will be none if failed 
        W = None
        embedding = None
    else:
        df = pd.DataFrame(eigenvecs, index = adata.obs_names, columns = np.round(np.arange(0,K,1),1))
        adata_evs = AnnData(df) #new feature representation
        sc.pp.neighbors(adata_evs, n_neighbors = k_embed) #find nn
        W = check_symmetric(adata_evs.obsp['connectivities']) #make symmetric if applicable
        embedding = np.asarray(df)

    return W, embedding

def precise_consensus(adata: AnnData,
                        x1key: str = 'Ms',
                        x2key: str = 'velocity',
                        n_pcs: int = 50,
                        n_pvs: int = 50,
                        random_state: int = 0,
                        k: int = 10,
                        X1 = None,
                        X2 = None,
                        logX1: bool = False,
                        logX2: bool = False,
                        **args):
    """merges data types through PRECISE: https://academic.oup.com/bioinformatics/article/35/14/i510/5529136
       implemented in python: https://github.com/NKI-CCB/PRECISE
       code is pulled directly from the github and modified to allow for most dissimilar principal vectors
       following merging, a kNN graph is constructed in joint PCA space

    Parameters
    adata: AnnData
        Annotated data object
    x1key: str (default = 'Ms')
        string referring to the layer of first matrix. Can be X, Ms, spliced, unspliced, velocity, or None
    x2key: str (default = 'velocity')
        string referring to the layer of second matrix. Can be X, Ms, spliced, unspliced, velocity, or None
    n_pcs: int (default = 50)
        integer referring to the number of principal components
    n_pvs: int (default = 50)
        integer referring to the number of principal vectors
            Positive values indicates most similar
            Negative indicates most dissimilar
    random_state: int (default = 0)
        integer referring to the random state for reproducibility 
    k: int (default = 10)
        integer referring to the number of nearest neighbors in k-nearest neighbor graph
    X1: (default = None)
        matrix referring to the first data type
    X2: (default = None)
        matrix referring to the second data type
    logX1: bool (default = False)
        boolean referring to whether the first data type should be log transformed
    logX2: bool (default = False)
        boolean referring to whether the second data type should be log transformed
    ----------

    Returns
    W: ndarray
        sparse symmetric matrix of connectivities from kNN graph in joint PC space. Dimensions are cells x cells
    embedding:
        joint PCA embedding. Dimensions are cells x n_pcs
    ----------
    """   
    X1, X2 = check_inputs(adata, x1key = x1key, x2key = x2key, X1 = X1, X2 = X2)
    X1, X2 = check_log(X1 = X1, X2 = X2, logX1 = logX1, logX2 = logX2)
    X1, X2 = check_sparse(X1 = X1, X2 = X2, return_sparse = False) #unfortunately needs to be dense here
    sys.stdout.write('X1 will be used as source. X2 will be used as target' + '\n')

    consensus = ConsensusRepresentation(source_data=X1,
                                        target_data=X2,
                                        n_factors=n_pcs,
                                        n_pv=n_pvs,
                                        dim_reduction='pca',
                                        n_representations=100,
                                        use_data=False,
                                        mean_center=True,
                                        std_unit=False)

    clf = PCA(n_components=n_pcs, random_state = random_state)
    
    Ps = clf.fit(X1).components_
    Ps = scipy.linalg.orth(Ps.transpose()).transpose()
    Pt = clf.fit(X2).components_
    Pt = scipy.linalg.orth(Pt.transpose()).transpose()

    u,_,v = np.linalg.svd(Ps.dot(Pt.transpose()))

    if n_pvs > 0:
        source_components_ = u.transpose().dot(Ps)[:n_pvs]
        target_components_ = v.dot(Pt)[:n_pvs]
    else:
        source_components_ = u.transpose().dot(Ps)[n_pvs:]
        target_components_ = v.dot(Pt)[n_pvs:]

    # Normalize to make sure that vectors are unitary
    source_components_ = normalize(source_components_, axis = 1)
    target_components_ = normalize(target_components_, axis = 1)

    initial_cosine_similarity_matrix_ = Ps.dot(Pt.transpose())
    cosine_similarity_matrix_ = source_components_.dot(target_components_.transpose())
    angles_ = np.arccos(np.diag(cosine_similarity_matrix_))

    principal_vectors = PVComputation(n_factors = n_pcs,
                                    n_pv = n_pvs,
                                    dim_reduction = 'pca',
                                    dim_reduction_target = 'pca')

    principal_vectors.source_components_ = source_components_.copy()
    principal_vectors.target_components_ = target_components_.copy()
    principal_vectors.initial_cosine_similarity_matrix_ = initial_cosine_similarity_matrix_.copy()
    principal_vectors.cosine_similarity_matrix_ = cosine_similarity_matrix_.copy()
    principal_vectors.angles_ = angles_.copy()

    t_sample = np.linspace(0, 1, 100 + 1)
    flow = np.array([IntermediateFactors._compute_flow_time(t, principal_vectors) for t in t_sample])
    flow_vectors = flow.transpose(1,0,2)
    
    consensus_representation = []
    for i in range(abs(n_pvs)):
        source_projected = flow_vectors[i].dot(X1.transpose())
        target_projected = flow_vectors[i].dot(X2.transpose())

        ks_stats = [
            ks_2samp(s,t)[0]
            for (s,t) in zip(np.asarray(source_projected), np.asarray(target_projected))
        ]

        consensus_representation.append(flow_vectors[i, np.argmin(ks_stats)])

    consensus_representation = np.array(consensus_representation).transpose()

    joint_pca = X1.dot(consensus_representation) #projects data on these factors 
    
    adata_joint = AnnData(adata.X)
    adata_joint.obs_names = adata.obs_names.values.copy()
    adata_joint.var_names = adata.var_names.values.copy()
    adata_joint.obsm['X_pca'] = np.asarray(joint_pca)
    scv.pp.neighbors(adata_joint, n_neighbors=k, n_pcs=abs(n_pvs)) #compute nn 

    W = adata_joint.obsp['connectivities'].copy()
    W = check_symmetric(W) #make symmetric if applicable
    embedding = np.asarray(adata_joint.obsm['X_pca'])

    return W, embedding

def precise(adata: AnnData,
            x1key: str = 'Ms',
            x2key: str = 'velocity',
            n_pcs: int = 50,
            n_pvs: int = 50,
            random_state: int = 0,
            k: int = 10,
            X1 = None,
            X2 = None,
            logX1: bool = False,
            logX2: bool = False,
            **args):
    """merges data types through PRECISE: https://academic.oup.com/bioinformatics/article/35/14/i510/5529136
       implemented in python: https://github.com/NKI-CCB/PRECISE
       code is from SpAGE: https://academic.oup.com/nar/article/48/18/e107/5909530, https://github.com/tabdelaal/SpaGE
       this isn't the full implementation of PRECISE. here, a kNN graph is constructed solely from projecting expression into expression PVs
       shared information is incorporated in similarity computation through dot product
       following projection, a kNN graph is constructed in PCA space

    Parameters
    adata: AnnData
        Annotated data object
    x1key: str (default = 'Ms')
        string referring to the layer of first matrix. Can be X, Ms, spliced, unspliced, velocity, or None
    x2key: str (default = 'velocity')
        string referring to the layer of second matrix. Can be X, Ms, spliced, unspliced, velocity, or None
    n_pcs: int (default = 50)
        integer referring to the number of principal components
    n_pvs: int (default = 50)
        integer referring to the number of principal vectors
            Positive values indicates most similar
            Negative indicates most dissimilar
    random_state: int (default = 0)
        integer referring to the random state for reproducibility 
    k: int (default = 10)
        integer referring to the number of nearest neighbors in k-nearest neighbor graph
    X1: (default = None)
        matrix referring to the first data type
    X2: (default = None)
        matrix referring to the second data type
    logX1: bool (default = False)
        boolean referring to whether the first data type should be log transformed
    logX2: bool (default = False)
        boolean referring to whether the second data type should be log transformed
    ----------

    Returns
    W: ndarray
        sparse symmetric matrix of connectivities from kNN graph in joint PC space. Dimensions are cells x cells
    embedding:
        joint PCA embedding. Dimensions are cells x n_pcs
    ----------
    """ 
    X1, X2 = check_inputs(adata, x1key = x1key, x2key = x2key, X1 = X1, X2 = X2)
    X1, X2 = check_log(X1 = X1, X2 = X2, logX1 = logX1, logX2 = logX2)
    X1, X2 = check_sparse(X1 = X1, X2 = X2, return_sparse = False)
    
    sys.stdout.write('X1 will be used as source. X2 will be used as target' + '\n')
    clf = PCA(n_components=n_pcs, random_state = random_state)
    Ps = clf.fit(X1).components_
    Ps = scipy.linalg.orth(Ps.transpose()).transpose()
    Pt =  clf.fit(X2).components_
    Pt = scipy.linalg.orth(Pt.transpose()).transpose()

    u,_,v = np.linalg.svd(Ps.dot(Pt.transpose()))
    
    if n_pvs > 0:
        source_components_ = u.transpose().dot(Ps)[:n_pvs]
        target_components_ = v.dot(Pt)[:n_pvs]
    else:
        source_components_ = u.transpose().dot(Ps)[n_pvs:]
        target_components_ = v.dot(Pt)[n_pvs:]
        
    source_components_ = normalize(source_components_, axis = 1)
    target_components_ = normalize(target_components_, axis = 1)

    PC_joint = X1.dot(source_components_.transpose())

    adata_joint = AnnData(adata.X)
    adata_joint.obs_names = adata.obs_names.values.copy()
    adata_joint.var_names = adata.var_names.values.copy()    
    adata_joint.obsm['X_pca'] = np.array(PC_joint)
    scv.pp.neighbors(adata_joint, n_neighbors=k, n_pcs=abs(n_pvs))
    
    W = adata_joint.obsp['connectivities'].copy()
    W = check_symmetric(W) #make symmetric if applicable
    embedding = np.asarray(adata_joint.obsm['X_pca'])

    return W, embedding

def integrated_diffusion(adata: AnnData,
                        x1key: str = 'Ms',
                        x2key: str = 'velocity',
                        k: int = 10,
                        k_embed: int = 10,
                        decay: int = 40,
                        distance: str = 'euclidean',
                        precomputed: str = None,
                        n_clusters: int = None,
                        random_state: int = 0,
                        t_max: int = 100,
                        K: int = 20,
                        X1 = None,
                        X2 = None,
                        logX1 = False,
                        logX2 = False,
                        n_jobs: int = 8,
                        **args):
    """merges data types through integrated diffusion: https://arxiv.org/pdf/2102.06757.pdf
       from integrated diffusion operator, powers and eigendecomposes

    Parameters
    adata: AnnData
        Annotated data object
    x1key: str (default = 'Ms')
        string referring to the layer of first matrix. Can be X, Ms, spliced, unspliced, velocity, or None
    x2key: str (default = 'velocity')
        string referring to the layer of second matrix. Can be X, Ms, spliced, unspliced, velocity, or None
    k: int (default = 10)
        integer referring to the number of nearest neighbors in k-nearest neighbor graph in kernel
    k_embed: int (default = 10)
        integer referring to the number of nearest neighbors in k-nearest neighbor graph in embedding
    decay: int (default = 40)
        rate of alpha decay to use
    distance: str (default = euclidean)
        distance metric for building kNN graph
    precomputed:
        string that denotes what type of graph if one is used as input. Can be either distance, affinity, adjacency, or None
    n_clusters: int (default = None)
        integer referring to the number of kmeans clusters for initialization
    random_state: int (default = 0)
        integer referring to the random state for reproducibility
    t_max: int (default = 100)
        maximum value of t to test
    K: int (default = 20)
        number of eigenvectors 
    X1: (default = None)
        matrix referring to the first data type
    X2: (default = None)
        matrix referring to the second data type
    logX1: bool (default = False)
        boolean referring to whether the first data type should be log transformed
    logX2: bool (default = False)
        boolean referring to whether the second data type should be log transformed
    n_jobs: int (default = 10)
        number of jobs to use in computation 
    ----------

    Returns
    W: ndarray
        joint embedding knn graph
    embedding:
        diffusion embedding for joint data
    ----------
    """  
    X1, X2 = check_inputs(adata, x1key = x1key, x2key = x2key, X1 = X1, X2 = X2)
    X1, X2 = check_log(X1 = X1, X2 = X2, logX1 = logX1, logX2 = logX2)
    X1, X2 = check_sparse(X1 = X1, X2 = X2, return_sparse = False) 

    #compute clusters for local denoising
    clusters_x1 = compute_phate_clusters(X = X1, k = k, decay = decay, n_jobs = n_jobs, n_clusters = n_clusters, random_state = random_state)
    sys.stdout.write('computed phate clusters for X1' +'\n')
    clusters_x2 = compute_phate_clusters(X = X2, k = k, decay = decay, n_jobs = n_jobs, n_clusters = n_clusters, random_state = random_state)
    sys.stdout.write('computed phate clusters for X2' +'\n')

    #perform local pca to denoise
    denoised_x1 = locally_denoise(X = X1, clusters = clusters_x1)
    sys.stdout.write('locally denoised X1' + '\n')
    denoised_x2 = locally_denoise(X = X2, clusters = clusters_x2)
    sys.stdout.write('locally denoised X2' + '\n')

    if (denoised_x1 is None) or (denoised_x2 is None):
        #this suggests too many clusters were chosen in local denoising. exit
        W = None
        embedding = None
        return W, embedding
    else:
        #compute diffusion operator for each data type
        p_x1 = compute_diff_aff(X = denoised_x1, k = k, decay = decay, distance = distance, precomputed = precomputed, n_jobs = n_jobs)
        sys.stdout.write('computed diffusion operator for X1' +'\n')
        p_x2 = compute_diff_aff(X = denoised_x2, k = k, decay = decay, distance = distance, precomputed = precomputed, n_jobs = n_jobs)
        sys.stdout.write('computed diffusion operator for X2' +'\n')
        
        p_mat = [p_x1, p_x2]
        
        t_opt = _find_t(p_mat, t_max = t_max)
        t_opt_reduced = _find_reduced_t(t_opt)
        
        #power data to reduced ratio and multiply for joint diffusion operator
        p_j = p_mat[0]**t_opt_reduced[0] @ p_mat[1]**t_opt_reduced[1]
        sys.stdout.write('computed joint diffusion operator' +'\n')
        h = compute_von_neumann_entropy(p_j.todense(), t_max = t_max)
        t_opt_j =  phate.vne.find_knee_point(y = h, x = np.arange(0, t_max))
        sys.stdout.write('joint t: {}'.format(str(t_opt_j)) +'\n')
        
        #power joint diffusion operator to obtain integrated diffusion operator
        P_j = p_j**t_opt_j 
        _, eigenvecs = eigendecomposition(W = P_j, K = K, which = 'LM') #largest mag because transition prob matrices 
        if eigenvecs is None:
            W = None
            embedding = None
        else:
            df = pd.DataFrame(eigenvecs, index = adata.obs_names, columns = np.round(np.arange(0,K,1),1))
            adata_evs = AnnData(df) #new feature representation
            sc.pp.neighbors(adata_evs, n_neighbors = k_embed) #find nn
            W = check_symmetric(adata_evs.obsp['connectivities']) #make symmetric if applicable
            embedding = np.asarray(df)

        return W, embedding

def grassmann(adata: AnnData,
                x1key: str = 'Ms',
                x2key: str = 'velocity',
                k: int = 10,
                k_embed: int = 10,
                K: int = None, 
                t: float = None,
                lam: float = 0.5,
                sym: str = 'max',
                normalized: bool = True,
                X1 = None,
                X2 = None,
                logX1 = False,
                logX2 = False,
                n_jobs: int = 10,
                return_adj: bool = False,
                **args):
    """merge data on grassmann manifold: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6513164/

    Parameters
    adata: AnnData
        Annotated data object
    x1key: str (default = 'Ms')
        string referring to the layer of first matrix. Can be X, Ms, spliced, unspliced, velocity, or None
    x2key: str (default = 'velocity')
        string referring to the layer of second matrix. Can be X, Ms, spliced, unspliced, velocity, or None 
    k: int (default = None)
        number of nearest neighbors to include for graph
    k_embed: int (default = 10)
        integer referring to the number of nearest neighbors in k-nearest neighbor graph in embedding
    K: int (default = 20)
        number of eigenvectors 
    t: int (default = None)
        integer referring to the scale of kernel bandwidth
    lam: float (default = 0.5)
        float referring to tradeoff between individual and shared subspaces
    sym: str (default = 'max')
        string referring to how to symmetrize the data.
    normalized: bool (default = None)
        whether to compute normalized or unnormalized graph laplacian
    X1: (default = None)
        matrix referring to the first data type
    X2: (default = None)
        matrix referring to the second data type
    logX1: bool (default = False)
        boolean referring to whether the first data type should be log transformed
    logX2: bool (default = False)
        boolean referring to whether the second data type should be log transformed
    n_jobs: int (default = 8)
        number of jobs to use for distance computation
    return_adj: bool (default = False)
        whether to return adjacency matrix. If False, a kNN graph is constructed from the embedding
    ----------

    Returns
    W: ndarray
        joint embedding knn graph
    embedding:
        laplacian embedding for joint data
    ----------
    """  
    X1, X2 = check_inputs(adata, x1key = x1key, x2key = x2key, X1 = X1, X2 = X2)
    X1, X2 = check_log(X1 = X1, X2 = X2, logX1 = logX1, logX2 = logX2)
    X1, X2 = check_sparse(X1 = X1, X2 = X2, return_sparse = False)
        
    X1_norm, X2_norm = norm(X1 = X1, X2 = X2) #normalize
    
    W_x1 = compute_grassmann_affinity(X1_norm, k = k, t = t, sym = sym, n_jobs = n_jobs)
    W_x2 = compute_grassmann_affinity(X2_norm, k = k, t = t, sym = sym, n_jobs = n_jobs)

    L_norm_x1 = evi.tl.compute_laplacian(W_x1, normalized = normalized) #compute normalized laplacian 
    L_norm_x2 = evi.tl.compute_laplacian(W_x2, normalized = normalized)
        
    _, eigenvecs_x1, = eigendecomposition(W = L_norm_x1, K = K, which = 'SM')
    _, eigenvecs_x2 = eigendecomposition(W = L_norm_x2, K = K, which = 'SM')
        
    if (eigenvecs_x1 is None) or (eigenvecs_x2 is None):
        W = None
        embedding = None
        return W, embedding
    else:
        #compute modified laplacian
        lmod = np.zeros(np.shape(W_x1))
        vecs = [eigenvecs_x1, eigenvecs_x2]
        for i in range(0, len(vecs)):
            lmod = lmod + vecs[i].dot(vecs[i].transpose())
        lmod = np.tril(lmod)+(np.tril(lmod, -1)).transpose()
    
        #combine and eigendecompose
        L_j = L_norm_x1 + L_norm_x2 - lam*lmod
        L_j = scipy.sparse.csr_matrix(L_j)
        eigenvals_j, eigenvecs_j = eigendecomposition(W = L_j, K = K, which = 'SM')

        #normalize
        eigenvecs_j_norm = normalize_evecs(eigenvecs_j)
        if eigenvecs_j_norm is None:
            W = None
            embedding = None
        else:
            df = pd.DataFrame(eigenvecs_j_norm, index = adata.obs_names, columns = np.round(np.arange(0,K,1),1))
            adata_evs = AnnData(df) #new feature representation
            sc.pp.neighbors(adata_evs, n_neighbors = k_embed) #find nn
            W = check_symmetric(adata_evs.obsp['connectivities']) #make symmetric if applicable
            embedding = np.asarray(df)
                
        return W, embedding


def check_inputs(adata: AnnData,
                x1key: str = 'Ms',
                x2key: str = 'velocity',
                X1 = None,
                X2 = None,
                **args):
    """Accesses matrices from adata object if keys are specified.

    Parameters
    adata: AnnData
        Annotated data object
    x1key: str (default = 'Ms')
        string referring to the layer of first matrix. Can be X, Ms, spliced, unspliced, velocity, or None
    x2key: str (default = 'velocity')
        string referring to the layer of second matrix. Can be X, Ms, spliced, unspliced, velocity, or None
    X1: (default = None)
        matrix referring to the first data type
    X2: (default = None)
        matrix referring to the second data type
    ----------

    Returns
    X1: Matrix
        matrix referring to the first data type
    X2: Matrix
        matrix referring to the second data type
    ----------
    """
    if X1 is None:
        if x1key == 'Ms':
            X1 = adata.layers['Ms'].copy()
        elif x1key == 'X':
            X1 = adata.X.copy()
        elif x1key == 'velocity':
            X1 = adata.layers['velocity'].copy()
        elif x1key == 'unspliced':
            X1 = adata.layers['unspliced'].copy()
        elif x1key == 'spliced':
            X1 = adata.layers['spliced'].copy()
        else:
            sys.stdout.write('X1 is not specified')
    
    if X2 is None:
        if x2key == 'Ms':
            X2 = adata.layers['Ms'].copy()
        elif x1key == 'X':
            X2 = adata.X.copy()
        elif x2key == 'velocity':
            X2 = adata.layers['velocity'].copy()
        elif x2key == 'unspliced':
            X2 = adata.layers['unspliced'].copy()
        elif x2key == 'spliced':
            X2 = adata.layers['spliced'].copy()
        else:
            sys.stdout.write('X2 is not specified')
            
    return X1, X2

def check_log(X1 = None,
                X2 = None,
                logX1: bool = False,
                logX2: bool = False,
                **args):
    """Converts matrices to log scale if specified.
        Important to note that following preprocessing:
                                X layer is log transformed
                                spliced and unspliced are not
                                Ms and velocity are derived matrices from log transformed data.
        Examples:
            if merging spliced and unspliced, log transform both
            if merging Ms and velocity, don't log transform
            if accessing X key for spliced, and merging with unspliced, only log transform unspliced
            
    Parameters
    X1: (default = None)
        matrix referring to the first data type
    X2: (default = None)
        matrix referring to the second data type
    logX1: bool (default = False)
        boolean referring to whether the first data type should be log transformed
    logX2: bool (default = False)
        boolean referring to whether the second data type should be log transformed
    ----------

    Returns
    X1: Matrix
        log transformed matrix of first data type if specified
    X2: Matrix
        log transformed matrix of second data type if specified
    ----------
    """
    if logX1 == True:
        if scipy.sparse.issparse(X1):
            X1 = scipy.sparse.csr_matrix.log1p(X1)
        else:
            X1 = np.log1p(X1)
    if logX2 == True:
        if scipy.sparse.issparse(X2):
            X2 = scipy.sparse.csr_matrix.log1p(X2)
        else:
            X2 = np.log1p(X2)
    return X1, X2

def check_sparse(X1 = None,
                X2 = None,
                return_sparse: bool = True,
                **args):
    """Converts data to sparse or dense format.

    Parameters
    X1: (default = None)
        matrix referring to the first data type
    X2: (default = None)
        matrix referring to the second data type
    return_sparse: bool (default = True)
        whether to convert matrices to sparse or to dense 
    ----------

    Returns
    X1: Matrix
        sparse or dense matrix referring to the first data type
    X2: Matrix
        sparse or dense matrix referring to the second data type
    ----------
    """
    if return_sparse == True:
        if scipy.sparse.issparse(X1) == False:
            X1 = scipy.sparse.csr_matrix(X1)
        if scipy.sparse.issparse(X2) == False:
            X2 = scipy.sparse.csr_matrix(X2)
    else:
        if scipy.sparse.issparse(X1) == True:
            X1 = X1.todense()
        if scipy.sparse.issparse(X2) == True:
            X2 = X2.todense()

    return X1, X2

def check_symmetric(W = None,
                    tol: float = 1e-8,
                    **args):
    """converts matrix to symmetric. If not sparse, converts to sparse

    Parameters
    W: Matrix
        adjacency, connectivity, transition probability matrix  
    tol: float (default = 1e-8)
        float referring to the tolerance of similarity in norm computation
    ----------

    Returns
    W: Matrix
        sparse symmetric matrix
    ----------
    """     
    if scipy.sparse.issparse(W):
        if scipy.sparse.linalg.norm(W-W.transpose(), scipy.Inf) < tol:
            sys.stdout.write('matrix is already symmetric'+'\n')
        else:
            sys.stdout.write('symmetrizing matrix'+'\n')
            W = (W + W.transpose()) / 2
    else:
        sys.stdout.write('converting to sparse'+'\n')
        W = scipy.sparse.csr_matrix(W)
        if scipy.sparse.linalg.norm(W-W.transpose(), scipy.Inf) < tol:
            sys.stdout.write('matrix is already symmetric'+'\n')
        else:
            sys.stdout.write('symmetrizing matrix'+'\n')
            W = (W + W.transpose()) / 2
    return W

def convert(adata: AnnData,
            xkey: str = 'X',
            X = None,
            W = None,
            embedding = None,
            cluster_key: str = None,
            clusters: list = None,
            **args):
    """prepares data for prediction.
            
    Parameters
    adata: AnnData
        Annotated data object
    xkey: str (default = X)
        string referring to the layer of expression data. Can be Ms or X
    X: (default = None)
        expression matrix
    W: (default = None)
        adjacency matrix of merged/ unmerged data
    embedding: (default = None)
        lower dimensional embedding 
    cluster_key: str (default = None)
        string referring to the cluster annotations in original adata object
    clusters: list (default = None)
        membership values for all cells 
    ----------

    Returns
    adata_pred: AnnData
        Annotated data object with graph of merged data
    ----------
    """    
    from scipy.spatial.distance import cdist

    if X is None:
        if xkey == 'Ms':
            X = adata.layers['Ms'].copy()
        else:
            X = adata.X.copy()

    if scipy.sparse.issparse(X) == False:
        X = scipy.sparse.csr_matrix(X)
    if scipy.sparse.issparse(W) == False:
        W = scipy.sparse.csr_matrix(W)

    adata_pred = AnnData(X)
    adata_pred.obs_names = adata.obs_names.copy()
    adata_pred.var_names = adata.var_names.copy()
        
    adata_pred.uns['neighbors'] = {'connectivities_key': 'connectivities'}
    adata_pred.uns['neighbors']['connectivities'] = W.copy()
    adata_pred.obsp['connectivities'] = W.copy()

    if embedding is not None:
        adata_pred.obsm['embedding'] = embedding.copy()

    if clusters is not None:
        adata_pred.obs['labels'] = pd.DataFrame(clusters)
    else:
        adata_pred.obs['labels'] = adata.obs[cluster_key].copy()
    
    return adata_pred

def norm(X1 = None,
        X2 = None,
        **args):
    """Mean center data with unit variance 

    Parameters
    X1: (default = None)
        matrix referring to the first data type
    X2: (default = None)
        matrix referring to the second data type
    ----------

    Returns
    X1: Matrix
        normalized matrix referring to the first data type
    X2: Matrix
        normalized matrix referring to the second data type
    ------
    """
    X1, X2 = check_sparse(X1 = X1, X2 = X2, return_sparse = False)
        
    stdscaler = StandardScaler()
    X1_norm = stdscaler.fit_transform(X = X1)
    X2_norm = stdscaler.fit_transform(X = X2)
    
    X1_norm = scipy.sparse.csr_matrix(X1_norm)
    X2_norm = scipy.sparse.csr_matrix(X2_norm)
    
    return X1_norm, X2_norm

def eigendecomposition(W = None,
                        K: int = None,
                        which: str = 'SM',
                        **args):
    """eigendecomposes matrix

    Parameters
    W: (default = None)
        matrix to eigendecompose
    K: int (default = None)
        number of eigenvectors 
    which: str (default = 'SM')
        which K eigenvectors and eigenvalues to find
            LM: largest magnitude
            SM: smallest magnitude
    ----------

    Returns
    eigenvals: ndarray
        array of K eigenvalues
    eigenvecs: ndarray
        array of K eigenvectors X[:, i]
    ----------
    """
    from scipy.sparse.linalg.eigen.arpack import ArpackNoConvergence, ArpackError
    
    if scipy.sparse.issparse(W) == False:
        W = scipy.sparse.csr_matrix(W)
    try:
        eigenvals,eigenvecs = scipy.sparse.linalg.eigs(W, k = K, which = which)
    except (ArpackNoConvergence, ArpackError) as error:
        sys.stderr.write('eigendecomposition failed to converge' + '\n')
        eigenvals = None
        eigenvecs = None
        
    if eigenvecs is not None:
        if np.isreal(eigenvecs).all():
            eigenvecs = eigenvecs.real
            eigenvals = eigenvals.real
        else:
            sys.stderr.write('eigendecomposition resulted in complex eigenvectors' + '\n')
            eigenvals = None
            eigenvecs = None
            
    return eigenvals, eigenvecs


def normalize_evecs(eigenvecs = None,
                    **args):
    """normalize eigenvectors

    Parameters
    eigenvecs: ndarray (default = None)
        eigenvectors to normalize
    ----------

    Returns
    eigenvecs_norm: ndarray
        array of normalized eigenvectors
    ----------
    """
    eigenvecs_norm = eigenvecs.copy()
    for i in range(0, np.shape(eigenvecs)[0]):
        if np.linalg.norm(eigenvecs[i,:]) != 0:
            eigenvecs_norm[i,:] = eigenvecs[i,:] / np.linalg.norm(eigenvecs[i,:])

    return eigenvecs_norm

def compute_phate_clusters(X = None,
                            k: int = 10,
                            decay: int = 40,
                            n_clusters: int = None,
                            random_state:int = 0,
                            n_jobs: int = 10,
                            **args):
    """computes kmeans clustering on phate operator

    Parameters
    X: (default = None)
        matrix referring to the data. Dimensions cells x genes 
    k: int (default = 10)
        integer referring to the number of nearest neighbors to build kernel
    decay: int (deafult = 40)
        sets decay rateof kernel tails for alpha decay
    n_clusters: int (default = None)
        integer referring to the number of kmeans clusters for initialization
    random_state: int (default = 0)
        integer for reproducibility
    n_jobs: int (default = 10)
        number of jobs to use
    ----------

    Returns
    clusters: ndarray
        membership value for every cell
    ----------
    """  
    phate_op = phate.PHATE(k = k, n_jobs = n_jobs, decay = decay, random_state = random_state)
    phate_op.fit(X)
    clusters = MiniBatchKMeans(n_clusters = n_clusters, random_state = random_state).fit_predict(phate_op.diff_potential)
    return clusters

def locally_denoise(X = None,
                    clusters = None,
                    **args):
    """locally denoises data based upon local PCA and truncated svd

    Parameters
    adata: AnnData
        Annotated data object
    X: (default = None)
        matrix referring to the data. Dimensions cells x genes 
    clusters: ndarray
        membership value for every cell
    ----------

    Returns
    denoised_x: ndarray
        matrix of denoised data. dimensions are cells x genes
    ----------
    """  
    denoised_x = np.zeros(np.shape(X))
    for i in np.unique(clusters):
        ind = np.where(clusters == i)[0]
        x = X[ind,:].copy()
        mean_c = scipy.sparse.csr_matrix.mean(x, axis=0)
        x_c = x - mean_c #subtract mean, ie mean center 
        U, s, Vh = np.linalg.svd(x_c, full_matrices=False) #svd
        try:
            knee_point = phate.vne.find_knee_point(s)
            s_matrix = np.diag(s[:knee_point])
            x_hat_uc = U[:,:knee_point].dot(s_matrix).dot(Vh[:knee_point,:]) #compute truncated svd
            x_hat =  x_hat_uc + mean_c #re-add mean
            denoised_x[ind, :] = x_hat.copy()
        except ValueError:
            sys.stdout.write('cannot find knee point. including all'+'\n')
            denoised_x[ind, :] = X[ind,:]

    denoised_x = scipy.sparse.csr_matrix(denoised_x)
    
    return denoised_x

def compute_diff_aff(X = None,
                        k: int = 10,
                        decay: int = 40,
                        distance: str = 'euclidean',
                        precomputed: str = None,
                        n_jobs: int = 10,
                        **args):
    """computes symmetric diffusion affinity matrix

    Parameters
    X: (default = None)
        matrix referring to the data. Dimensions cells x genes
    k: int (default = 30)
        number of nearest neighbors to build the graph
    decay: int (default = 40)
        rate of alpha decay to use
    distance: str (default = euclidean)
        distance metric for building kNN graph
    precomputed:
        string that denotes what type of graph if one is used as input. Can be either distance, affinity, adjacency, or None
    n_jobs: int (default = 10)
        number of jobs to use in computation
    ----------

    Returns
    diff_aff: ndarray
        symmetric diffusion affinity matrix. Dimensions cells x cells
    ----------
    """  
    g = graphtools.api.Graph(X, knn = k, decay = decay, distance = distance, precomputed = precomputed, n_jobs = n_jobs)
    diff_aff = g.diff_aff.copy() #symmetric diffusion operator
    return diff_aff

def compute_von_neumann_entropy(data, t_max: int = 100, **args):
    """computes von neumann entropy using phate vne
       This code is pulled from phate directly: https://github.com/KrishnaswamyLab/PHATE/blob/33846d3a1580af0714a740da7df77fd90271c592/Python/phate/vne.py
       Altered for quick computation, where full_matrices is set to False
       
    Parameters
    data: (default = None)
        matrix referring to the data. Dimensions cells x genes
    t_max: int (default = 100)
        maximum value of t to test
    ----------

    Returns
    entropy: ndarray
        array containing entropy values for all values of t
    ----------
    """  
    _, eigenvalues, _ = np.linalg.svd(data, full_matrices = False,**args)
    entropy = []
    eigenvalues_t = np.copy(eigenvalues)
    for _ in range(t_max):
        prob = eigenvalues_t / np.sum(eigenvalues_t)
        prob = prob + np.finfo(float).eps
        entropy.append(-np.sum(prob * np.log(prob)))
        eigenvalues_t = eigenvalues_t * eigenvalues
    entropy = np.array(entropy)
    
    return np.array(entropy)

def heat_kernel(X, t):
    s = np.exp(- (X*X)/ (2.*t**2))
    return s

def compute_grassmann_affinity(X = None,
                                k: int = 10,
                                t: int = None,
                                sym: str = 'max',
                                n_jobs: int = -1,
                                **args):
    """compute grassmann affinity matrix using heat kernel

    Parameters
    X: (default = None)
        matrix referring to the data. Dimensions cells x genes 
    k: int (default = None)
        number of nearest neighbors to include for graph
    t: int (default = None)
        integer referring to the scale of kernel bandwidth
    sym: str (default = 'max')
        string referring to how to symmetrize the data.
    n_jobs: int (default = 8)
        number of jobs to use for distance computation
    ----------

    Returns
    W: ndarray
        symmetric sparse affinity matrix. Dimensions are cells x cells
    ----------
    """  
    dist_mat = sklearn.metrics.pairwise.pairwise_distances(X, X, metric='euclidean', n_jobs = n_jobs) #this is a bad choice and should be sped up
    nn = np.argsort(dist_mat, axis = 1)[:,1:k+1]
    
    _W = []
    for i in range(X.shape[0]):
        w_ = np.zeros((1, X.shape[0]))
        s = np.array([heat_kernel(dist_mat[i,v], t) for v in nn[i]] )
        np.put(w_, nn[i], s)
        _W.append(w_[0])

    _W = np.array(_W)
        
    if sym == 'max':
        W = np.maximum(_W, _W.transpose()) #make symmetric. defaults to max, as it was this in their code
    else:
        W = (_W + _W.transpose()) / 2
    
    W = scipy.sparse.csr_matrix(W)
        
    return W

def _find_t(P = None,
            t_max: int = 100,
            **args):
    """finds intrinsic dimensionality of the data through spectral entropy of transition matrix

    Parameters
    P: (default = None)
        transition probability matrix. [P]
    t_max: int (default = 100)
        maximum value of t to test
    ----------

    Returns
    t_opt: ndarray
        intrinsic dimensionality of each data type 
    ----------
    """      
    t = np.arange(0, t_max)
    t_opt = []
    for i in range(0, len(P)):
        p = P[i]
        h = compute_von_neumann_entropy(p.todense(), t_max = t_max) #needs to be dense
        t_opt.append(phate.vne.find_knee_point(y = h, x = t))

    t_opt = np.asarray(t_opt)
    sys.stdout.write('optimal t: {}'.format(str(t_opt)) + '\n')
    return np.asarray(t_opt)

def _find_reduced_t(t_opt, **args):
    """finds intrinsic dimensionality of the data through spectral entropy of transition matrix

    Parameters
    t_opt: ndarray
        intrinsic dimensionality of each data type 
    ----------

    Returns
    t_opt_reduced: ndarray
        reduced ratio of information of each data type
    ----------
    """      
    #this is beyond gross, but will rewrite one day. i'm sorry if you're reading it and i forgot to change it :) 
    t_opt_reduced = (t_opt / np.gcd.reduce(t_opt)).astype(int)
    
    if np.array_equal(t_opt_reduced, t_opt):
        t_p1 = ((t_opt_reduced+1) / np.gcd.reduce(t_opt_reduced+1)).astype(int)
        t1_p1 = ((np.array([t_opt_reduced[0] + 1, t_opt_reduced[1]])) / np.gcd.reduce(np.array([t_opt_reduced[0] + 1, t_opt_reduced[1]]))).astype(int)
        t2_p1 = ((np.array([t_opt_reduced[0], t_opt_reduced[1] + 1])) / np.gcd.reduce(np.array([t_opt_reduced[0], t_opt_reduced[1] + 1]))).astype(int)
        t_m1 = ((t_opt_reduced-1) / np.gcd.reduce(t_opt_reduced-1)).astype(int)
        t1_m1 = ((np.array([t_opt_reduced[0] - 1, t_opt_reduced[1]])) / np.gcd.reduce(np.array([t_opt_reduced[0] - 1, t_opt_reduced[1]]))).astype(int)
        t2_m1 = ((np.array([t_opt_reduced[0], t_opt_reduced[1] - 1])) / np.gcd.reduce(np.array([t_opt_reduced[0], t_opt_reduced[1] - 1]))).astype(int)

        idx = [t_p1, t1_p1, t2_p1, t_m1, t1_m1, t2_m1]
        t_opt_reduced = idx[np.argmin(np.sum(idx, axis = 1))] #finds smallest reduced ratio
        sys.stdout.write('approximate reduced ratio t: {}'.format(str(t_opt_reduced)) + '\n')  
    else:
        sys.stdout.write('reduced ratio t: {}'.format(str(t_opt_reduced)) + '\n')
    return t_opt_reduced
