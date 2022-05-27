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
import os
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import anndata2ri

pandas2ri.activate()
anndata2ri.activate()
import time

##series of scripts only for evaluating runtime scalability

def concat_merge_rt(adata: AnnData,
                x1key: str = 'Ms',
                x2key: str = 'velocity',
                X1 = None,
                X2 = None,
                logX1: bool = False,
                logX2: bool = False,
                **args):

    X1, X2 = evi.tl.check_inputs(adata, x1key = x1key, x2key = x2key, X1 = X1, X2 = X2)
    X1, X2 = evi.tl.check_log(X1 = X1, X2 = X2, logX1 = logX1, logX2 = logX2)
    X1, X2 = evi.tl.check_sparse(X1 = X1, X2 = X2, return_sparse = True)

    tic = time.perf_counter()
    X_merge = scipy.sparse.hstack((X1, X2)) #n x 2*p
    toc = time.perf_counter()

    elapsed = toc-tic

    return elapsed

def sum_merge_rt(adata: AnnData,
                x1key: str = 'Ms',
                x2key: str = 'velocity',
                X1 = None,
                X2 = None,
                logX1: bool = False,
                logX2: bool = False,
                **args):

    X1, X2 = evi.tl.check_inputs(adata, x1key = x1key, x2key = x2key, X1 = X1, X2 = X2)
    X1, X2 = evi.tl.check_log(X1 = X1, X2 = X2, logX1 = logX1, logX2 = logX2)
    X1, X2 = evi.tl.check_sparse(X1 = X1, X2 = X2, return_sparse = True)

    tic = time.perf_counter()
    X_merge = X1 + X2 #n x p
    toc = time.perf_counter()
    elapsed = toc-tic

    return elapsed

def cellrank_rt(adata: AnnData,
            x1key: str = 'Ms',
            x2key: str = 'velocity',
            lam: float = 0.2,
            scheme: str = 'correlation',
            mode: str = 'deterministic',
            **args):

    from cellrank.tl.kernels import VelocityKernel, ConnectivityKernel

    if x2key != 'velocity':
        sys.stderr.write('x2key must be velocity. respecify')
        pass

    adata_pp = adata.copy()
    tic = time.perf_counter()
    sc.pp.pca(adata_pp, n_comps=50)
    sc.pp.neighbors(adata_pp, n_neighbors = 10)

    vk = VelocityKernel(adata_pp, xkey = x1key, vkey = x2key).compute_transition_matrix(scheme = scheme, mode = mode) #n x n 
    ck = ConnectivityKernel(adata_pp).compute_transition_matrix() #n x n 
    combined_kernel = (((1-lam) * ck) + (lam * vk)).compute_transition_matrix()
    toc = time.perf_counter()
    elapsed = toc-tic
    
    return elapsed

def snf_rt(adata: AnnData,
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
    from snf import compute

    X1, X2 = evi.tl.check_inputs(adata, x1key = x1key, x2key = x2key, X1 = X1, X2 = X2)
    X1, X2 = evi.tl.check_log(X1 = X1, X2 = X2, logX1 = logX1, logX2 = logX2)
    X1, X2 = evi.tl.check_sparse(X1 = X1, X2 = X2, return_sparse = False) #requires dense :(
              
    C = [X1, X2]
    tic = time.perf_counter()
    adj_i = compute.make_affinity(C, metric = metric, K = k, mu = mu) #2*n x n
    adj_j = compute.snf(adj_i, K = k) #n x n
    toc = time.perf_counter()
    elapsed = toc-tic

    return elapsed

def precise_consensus_rt(adata: AnnData,
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
    X1, X2 = evi.tl.check_inputs(adata, x1key = x1key, x2key = x2key, X1 = X1, X2 = X2)
    X1, X2 = evi.tl.check_log(X1 = X1, X2 = X2, logX1 = logX1, logX2 = logX2)
    X1, X2 = evi.tl.check_sparse(X1 = X1, X2 = X2, return_sparse = False) #unfortunately needs to be dense here
    sys.stdout.write('X1 will be used as source. X2 will be used as target' + '\n')

    tic = time.perf_counter()

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
    toc = time.perf_counter()
    elapsed = toc-tic

    return elapsed

def precise_rt(adata: AnnData,
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

    X1, X2 = evi.tl.check_inputs(adata, x1key = x1key, x2key = x2key, X1 = X1, X2 = X2)
    X1, X2 = evi.tl.check_log(X1 = X1, X2 = X2, logX1 = logX1, logX2 = logX2)
    X1, X2 = evi.tl.check_sparse(X1 = X1, X2 = X2, return_sparse = False)

    tic = time.perf_counter()
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
    toc = time.perf_counter()
    elapsed = toc-tic

    return elapsed

def integrated_diffusion_rt(adata: AnnData,
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
                        n_jobs: int = 1,
                        **args):
    X1, X2 = evi.tl.check_inputs(adata, x1key = x1key, x2key = x2key, X1 = X1, X2 = X2)
    X1, X2 = evi.tl.check_log(X1 = X1, X2 = X2, logX1 = logX1, logX2 = logX2)
    X1, X2 = evi.tl.check_sparse(X1 = X1, X2 = X2, return_sparse = False) 

    #compute clusters for local denoising
    tic = time.perf_counter()
    clusters_x1 = evi.tl.compute_phate_clusters(X = X1, k = k, decay = decay, n_jobs = n_jobs, n_clusters = n_clusters, random_state = random_state)
    clusters_x2 = evi.tl.compute_phate_clusters(X = X2, k = k, decay = decay, n_jobs = n_jobs, n_clusters = n_clusters, random_state = random_state)

    #perform local pca to denoise
    denoised_x1 = evi.tl.locally_denoise(X = X1, clusters = clusters_x1)
    denoised_x2 = evi.tl.locally_denoise(X = X2, clusters = clusters_x2)
    
    p_x1 = evi.tl.compute_diff_aff(X = denoised_x1, k = k, decay = decay, distance = distance, precomputed = precomputed, n_jobs = n_jobs)
    p_x2 = evi.tl.compute_diff_aff(X = denoised_x2, k = k, decay = decay, distance = distance, precomputed = precomputed, n_jobs = n_jobs)
        
    p_mat = [p_x1, p_x2]
        
    t_opt = evi.tl.merge._find_t(p_mat, t_max = t_max)
    t_opt_reduced = evi.tl.merge._find_reduced_t(t_opt)
        
    p_j = p_mat[0]**t_opt_reduced[0] @ p_mat[1]**t_opt_reduced[1]
    h = evi.tl.compute_von_neumann_entropy(p_j.todense(), t_max = t_max)
    t_opt_j =  phate.vne.find_knee_point(y = h, x = np.arange(0, t_max))
    P_j = p_j**t_opt_j 
    toc = time.perf_counter()
    elapsed = toc-tic

    return elapsed

def grassmann_rt(adata: AnnData,
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
                n_jobs: int = 1,
                return_adj: bool = False,
                **args):

    X1, X2 = evi.tl.check_inputs(adata, x1key = x1key, x2key = x2key, X1 = X1, X2 = X2)
    X1, X2 = evi.tl.check_log(X1 = X1, X2 = X2, logX1 = logX1, logX2 = logX2)
    X1, X2 = evi.tl.check_sparse(X1 = X1, X2 = X2, return_sparse = False)

    tic = time.perf_counter()    
    X1_norm, X2_norm = evi.tl.merge.norm(X1 = X1, X2 = X2) #normalize
    
    W_x1 = evi.tl.compute_grassmann_affinity(X1_norm, k = k, t = t, sym = sym, n_jobs = n_jobs)
    W_x2 = evi.tl.compute_grassmann_affinity(X2_norm, k = k, t = t, sym = sym, n_jobs = n_jobs)

    L_norm_x1 = evi.tl.compute_laplacian(W_x1, normalized = normalized) #compute normalized laplacian 
    L_norm_x2 = evi.tl.compute_laplacian(W_x2, normalized = normalized)
        
    _, eigenvecs_x1, = evi.tl.eigendecomposition(W = L_norm_x1, K = K, which = 'SM')
    _, eigenvecs_x2 = evi.tl.eigendecomposition(W = L_norm_x2, K = K, which = 'SM')

    lmod = np.zeros(np.shape(W_x1))
    vecs = [eigenvecs_x1, eigenvecs_x2]
    for i in range(0, len(vecs)):
        lmod = lmod + vecs[i].dot(vecs[i].transpose())
    lmod = np.tril(lmod)+(np.tril(lmod, -1)).transpose()

    #combine and eigendecompose
    L_j = L_norm_x1 + L_norm_x2 - lam*lmod
    L_j = scipy.sparse.csr_matrix(L_j)
    eigenvals_j, eigenvecs_j = evi.tl.eigendecomposition(W = L_j, K = K, which = 'SM')

    #normalize
    eigenvecs_j_norm = evi.tl.normalize_evecs(eigenvecs_j)
    toc = time.perf_counter()
    elapsed = toc-tic
    return elapsed

def mofap_rt(adata: AnnData,
        x1key: str = 'Ms',
        x2key: str = 'velocity',
        X1 = None,
        X2 = None,
        logX1: bool = False,
        logX2: bool = False,
        K: int = 20,
        k: int = 10, 
        random_state: int = 0,
        groups_key = None,
        gpu = False,
        **args):

    from mofapy2.run.entry_point import entry_point
    
    adata.obs_names_make_unique() #if some cell names are the same, this will throw an error in MOFA as giving duplicates when they aren't
    adata.var_names_make_unique()

    X1, X2 = evi.tl.check_inputs(adata, x1key = x1key, x2key = x2key, X1 = X1, X2 = X2)
    X1, X2 = evi.tl.check_log(X1 = X1, X2 = X2, logX1 = logX1, logX2 = logX2)
    X1, X2 = evi.tl.check_sparse(X1 = X1, X2 = X2, return_sparse = False)

    df1 = pd.DataFrame(X1, index = adata.obs_names, columns = adata.var_names)
    df2 = pd.DataFrame(X2, index = adata.obs_names, columns = adata.var_names)

    if groups_key is None:
        groups = 'group_0'
    else:
        groups = adata.obs[groups_key]

    df1 = pd.DataFrame(X1, index = adata.obs_names, columns = adata.var_names)
    df1['group'] = groups
    df1['view'] = x1key
    df2 = pd.DataFrame(X2, index = adata.obs_names, columns = adata.var_names)
    df2['group'] = groups
    df2['view'] = x2key

    df1 = df1.melt(ignore_index = False, id_vars = ['group', 'view'])
    df2 = df2.melt(ignore_index = False, id_vars = ['group', 'view'])

    df = pd.concat([df1, df2], axis = 0, ignore_index = False)
    df.reset_index(inplace = True)

    df.rename(columns={"index": "sample", "Gene": "feature", "value": "value", "view": "view", "group": "group"}, inplace = True)
    
    tic = time.perf_counter() 

    ent = entry_point()
    ent.set_data_options(scale_groups = False, scale_views = True)
    ent.set_data_df(df)
    ent.set_model_options(factors = K, spikeslab_weights = True)
    ent.set_train_options(iter = 500, convergence_mode = "fast", startELBO = 1, freqELBO = 1, gpu_mode = gpu, verbose = False, seed = random_state)

    ent.build()
    ent.run()
    toc = time.perf_counter()
    elapsed = toc-tic

    return elapsed

def seurat_v4_rt(adata: AnnData,
            x1key: str = 'Ms',
            x2key: str = 'velocity',
            X1 = None,
            X2 = None,
            logX1 = False,
            logX2 = False,
            k: int = 10,
            n_pcs: int = 50,
            **args):

    adata.obs_names_make_unique() #make cell names different if they aren't already
    adata.var_names_make_unique()
    X1, X2 = evi.tl.check_inputs(adata, x1key = x1key, x2key = x2key, X1 = X1, X2 = X2)
    X1, X2 = evi.tl.check_log(X1 = X1, X2 = X2, logX1 = logX1, logX2 = logX2)
    X1, X2 = evi.tl.check_sparse(X1 = X1, X2 = X2, return_sparse = False)

    tic = time.perf_counter()

    adata_x1 = AnnData(X1)
    adata_x2 = AnnData(X2)

    sc.tl.pca(adata_x1, n_comps = n_pcs)
    sc.tl.pca(adata_x2, n_comps = n_pcs)

    X1 = pd.DataFrame(X1, index = adata.obs_names, columns = adata.var_names)
    X2 = pd.DataFrame(X2, index = adata.obs_names, columns = adata.var_names)
    X1_pca = pd.DataFrame(adata_x1.obsm['X_pca'], index = adata.obs_names)
    X2_pca = pd.DataFrame(adata_x2.obsm['X_pca'], index = adata.obs_names)
    toc = time.perf_counter()
    elapsed_1 = toc-tic

    X1.to_csv('X1.csv')
    X2.to_csv('X2.csv')
    X1_pca.to_csv('X1_pca.csv')
    X2_pca.to_csv('X2_pca.csv')

    r = robjects.r
    r['source'](os.path.join('evi', 'tools', 'runtime.R'))
    seurat_v4_rt_r = robjects.globalenv['seurat_v4_rt']

    result_r = seurat_v4_rt_r(k)
    elapsed_2 = result_r

    elapsed = elapsed_1 + elapsed_2 

    os.remove('X1.csv')
    os.remove('X2.csv')
    os.remove('X1_pca.csv')
    os.remove('X2_pca.csv')

    return elapsed