import os
import sys
import evi
from anndata import AnnData
import scvelo as scv
import scanpy as sc
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import anndata2ri

pandas2ri.activate()
anndata2ri.activate()

def annotate_Schafflick(adata: AnnData,
                        sample: str = None):
    """Annotates Schafflick dataset with cell names, patient id, sample type, and disease status.

    Parameters
    adata: AnnData
        Annotated data object
    ----------

    Returns
    adata: AnnData
        Annotated data object with appended condition information.
    ----------
    """
    cell_names = [i.split(':')[1] for i in adata.obs_names.values]
    patient = [i.split('_')[1] for i in adata.obs_names.values]
    sample = [i.split('_')[2] for i in adata.obs_names.values]

    status = np.repeat('NA', len(patient)).astype('object')
    idx_MS = [i for i, si in enumerate(patient) if si.startswith('MS')] #MS patients have MS identifier
    idx_control = [i for i, si in enumerate(patient) if si.startswith('P')] #control patients start with P
    status[idx_MS] = 'MS'
    status[idx_control] = 'control'

    adata.obs_names = cell_names.copy() #cell barcodes truncated with -# removed so make unique
    adata.obs_names_make_unique()
    adata.obs['patient'] = patient.copy()
    adata.obs['sample'] = sample.copy()
    adata.obs['status'] = status.copy() #can make as a copy 
    
    return adata

def annotate_stats(adata: AnnData,
                   mitostr: str = 'MT-'):
    """Annotates summary count, gene, and mitochondrial percentage.

    Parameters
    adata: AnnData
        Annotated data object
    mitostr: str
        string specifying mitochondarial genes. MT- for human, mt- for mouse
    ----------

    Returns
    adata: AnnData
        Annotated data object with appended summary information:
            adata.obs[n_counts]
            adata.obs[n_genes]
            adata.obs['mito_ratio']
    ----------
    """
    adata.obs['n_counts'] = pd.DataFrame((adata.X.sum(1).flatten().tolist()[0]), index = adata.obs_names)
    adata.obs['n_genes'] = pd.DataFrame((adata.X > 0).sum(1).flatten().tolist()[0], index = adata.obs_names)
    mito_genes = adata.var_names.str.startswith(mitostr)
    pct_mito = np.sum(adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
    adata.obs['mito_ratio'] = pd.DataFrame(pct_mito.tolist(), index = adata.obs_names).values
    
    return adata

def filter_cells_genes(adata: AnnData,
                        min_counts: int = None,
                        max_counts: int = None,
                        min_genes: int = None,
                        max_genes: int = None,
                        min_shared_counts: int = None,
                        min_shared_cells: int = None,
                        mito_ratio: float = None):
    """Filters adata object according to cell and gene criteria.

    Parameters
    adata: AnnData
        Annotated data object
    min_counts: int (default = None)
        integer referring to the minimum number of gene expression counts (dropout)
    max_counts: int (default = None)
        integer referring to the maximum number of gene expression counts (doublet)
    min_genes: int (default = None)
        integer referring to the minimum number of genes required for a cell (low quality)
    max_genes: int (default = None)
        integer referring to the maximum number of genes required for a cell
    min_shared_counts: int (default = None)
        integer referring to the minimum number of unspliced and spliced counts required for a gene
    min_shared_cells: int (default = None)
        integer referring to the number of cells required to be expressed in both spliced and unspliced
    mito_ratio: float (default = None)
        float referring to the ratio of mitochondrial transcripts. keep > ratio.
    ----------

    Returns
    adata: AnnData
        filtered annotated adata object
    ----------
    """
    print('Total number of cells: {:d}'.format(adata.n_obs))

    if max_counts is not None:
        sc.pp.filter_cells(adata, max_counts = max_counts)
    if min_counts is not None:
        sc.pp.filter_cells(adata, min_counts = min_counts)
    if min_genes is not None:
        sc.pp.filter_cells(adata, min_genes = min_genes)
    if max_genes is not None:
        sc.pp.filter_cells(adata, max_genes = max_genes)

    print('Number of cells after max,min count filter and max, min gene filter: {:d}'.format(adata.n_obs))
    
    if mito_ratio is not None:
        adata = adata[adata.obs['mito_ratio'] < mito_ratio]
        print('Number of cells after MT filter: {:d}'.format(adata.n_obs))
    
    scv.pp.filter_genes(adata, min_shared_counts = min_shared_counts, min_shared_cells = min_shared_cells)
    print('Number of genes after min_shared_counts and min_shared_cells filter: {:d}'.format(adata.n_vars))
    
    print('dimensions post filtering', adata.shape)

    return adata

def run_multiBatchNorm(adata: AnnData,
                        X: list = None,
                        batches: list = None,
                        clusters: list = None,
                        n_pcs: int = 50,
                        k: int = 10,
                        resolution: float = 0.5):
    """performs per-batch scaling normalization with batchelor: https://www.nature.com/articles/nbt.4091
       implemented in R: https://rdrr.io/github/LTLA/batchelor/man/multiBatchNorm.html

    Parameters
    adata: AnnData
        Annotated data object
    X: list (default = None)
        list of spliced and unspliced counts. 
            If provided: dimensions must be [(genes, cells), (genes, cells)].
            If not provided: X, unspliced layers will be used
    batches: list (default = None)
        list referring to batch condition of every cell. Dimensions must be (1 x cells) 
    clusters: list (default = None)
        list referring to cluster annotation for every cell. Dimensions must be (1 x cells) 
    n_pcs: int (default = 50)
        integer referring to the number of principal components
            if == 0, use all data for nearest neighbors
    k: int (default = 10)
        integrer referring to the number of nearest neighbors
    resolution: float (default = 0.5)
        float referring to the resolution parameter for leiden clustering
    ----------

    Returns
    adata: AnnData
        batch normalized annotated adata object, where x and spliced factors are the same. 
            adata.obs['size_factors_x']
            adata.obs['size_factors_s']
            adata.obs['size_factors_u']

    ----------
    """
    if X is None:
        X = [adata.X.todense().transpose(), adata.layers['unspliced'].todense().transpose()]
    elif scipy.sparse.issparse(X[0]) or scipy.sparse.issparse(X[1]):
        X = [X[0].todense(), X[1].todense()]

    if clusters is None:
        clusters = _compute_clusters(adata, log_transform = True, n_pcs = n_pcs, k = k, resolution = resolution)
        
    r = robjects.r
    r['source'](os.path.join('evi','preprocessing','preprocess.R'))
    run_multiBatchNorm_r = robjects.globalenv['run_multiBatchNorm']

    size_factors = run_multiBatchNorm_r(X1 = pandas2ri.py2rpy(pd.DataFrame(X[0]).astype('int64')),
                                        X2 = pandas2ri.py2rpy(pd.DataFrame(X[1]).astype('int64')),
                                        batches = pandas2ri.py2rpy(pd.DataFrame(batches).transpose()),
                                        clusters = pandas2ri.py2rpy(pd.DataFrame(clusters.astype('str').values).transpose()))
    
    adata.obs['size_factors_x'] = size_factors[0].transpose()
    adata.obs['size_factors_s'] = size_factors[0].transpose()
    adata.obs['size_factors_u'] = size_factors[1].transpose()

    sc.pl.scatter(adata, 'size_factors_x', 'n_counts')
    sc.pl.scatter(adata, 'size_factors_x', 'n_genes')
    
    sys.stdout.write('performing normalization with size factors'+'\n')
    adata.X = adata.X.todense() / adata.obs['size_factors_x'].values[:, None]
    adata.layers['spliced'] = adata.layers['spliced'].todense() / adata.obs['size_factors_s'].values[:, None]
    adata.layers['unspliced'] = adata.layers['unspliced'].todense() / adata.obs['size_factors_u'].values[:, None]

    adata.X = scipy.sparse.csr_matrix(adata.X) #convert to sparse or hvg will fail
    adata.layers['spliced'] = scipy.sparse.csr_matrix(adata.layers['spliced'])
    adata.layers['unspliced'] = scipy.sparse.csr_matrix(adata.layers['unspliced'])
    
    return adata

def run_scran(adata: AnnData,
                X: list = None,
                clusters: list = None,
                n_pcs: int = 50,
                k: int = 10,
                resolution: float = 0.5):
    """performs normalization with scran: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-0947-7
       implemented in R: https://rdrr.io/bioc/scran/man/computeSumFactors.html

    Parameters
    adata: AnnData
        Annotated data object
    X: list = None
        list of spliced and unspliced counts. 
            If provided: dimensions must be [(genes, cells), (genes, cells)].
            If not provided: X, unspliced layers will be used
    clusters: list (default = None)
        list referring to cluster annotation for every cell. Dimensions must be (1 x cells) 
    n_pcs: int (default = 50)
        integer referring to the number of principal components
            if == 0, use all data for nearest neighbors
    k: int (default = 10)
        integrer referring to the number of nearest neighbors
    resolution: float (default = 0.5)
        float referring to the resolution parameter for leiden clustering
    ----------

    Returns
    adata: AnnData
        batch normalized annotated adata object, where x and spliced factors are the same. 
            adata.obs['size_factors_x']
            adata.obs['size_factors_s']
            adata.obs['size_factors_u']

    ----------
    """
    if X is None:
        X = [adata.X.todense().transpose(), adata.layers['unspliced'].todense().transpose()]
    elif scipy.sparse.issparse(X[0]) or scipy.sparse.issparse(X[1]):
        X = [X[0].todense(), X[1].todense()]

    if clusters is None:
        clusters = _compute_clusters(adata, log_transform = True, n_pcs = n_pcs, k = k, resolution = resolution)

    r = robjects.r
    r['source'](os.path.join('evi','preprocessing','preprocess.R'))
    run_scran_r = robjects.globalenv['run_scran']
    
    size_factors = run_scran_r(X1 = pandas2ri.py2rpy(pd.DataFrame(X[0]).astype('int64')),
                                X2 = pandas2ri.py2rpy(pd.DataFrame(X[1]).astype('int64')),
                                clusters = pandas2ri.py2rpy(pd.DataFrame(clusters.astype('str').values).transpose()))
    
    adata.obs['size_factors_x'] = size_factors[0].transpose()
    adata.obs['size_factors_s'] = size_factors[0].transpose()
    adata.obs['size_factors_u'] = size_factors[1].transpose()

    sc.pl.scatter(adata, 'size_factors_x', 'n_counts')
    sc.pl.scatter(adata, 'size_factors_x', 'n_genes')
    
    sys.stdout.write('performing normalization with size factors'+'\n')
    adata.X = adata.X.todense() / adata.obs['size_factors_x'].values[:, None]
    adata.layers['spliced'] = adata.layers['spliced'].todense() / adata.obs['size_factors_s'].values[:, None]
    adata.layers['unspliced'] = adata.layers['unspliced'].todense() / adata.obs['size_factors_u'].values[:, None]

    adata.X = scipy.sparse.csr_matrix(adata.X) #convert to sparse or hvg will fail
    adata.layers['spliced'] = scipy.sparse.csr_matrix(adata.layers['spliced'])
    adata.layers['unspliced'] = scipy.sparse.csr_matrix(adata.layers['unspliced'])
    
    return adata

def _compute_clusters(adata: AnnData,
                     log_transform: bool = True,
                     n_pcs: int = 50,
                     k: int = 10,
                     resolution: float = 0.5):
    """computes leiden clustering specifically for normalization strategies

    Parameters
    adata: AnnData
        Annotated data object
    log_transform: bool (default = True)
        Boolean indicating whether log transformation needs to be performed 
    n_pcs: int (default = 50)
        integer referring to the number of principal components
            if == 0, use all data for nearest neighbors
    k: int (default = 10)
        integrer referring to the number of nearest neighbors
    resolution: float (default = 0.5)
        float referring to the resolution parameter for leiden clustering
    ----------

    Returns
    clusters: series object of clusters

    ----------
    """
    adata_pp = adata.copy()
    if log_transform == True:
        sc.pp.log1p(adata_pp)
    if n_pcs == 0:
        scv.pp.neighbors(adata_pp, n_neighbors=k, n_pcs=0)
    else:
        sc.tl.pca(adata_pp, n_comps = n_pcs, zero_center = True, svd_solver = 'arpack', random_state = 0) #random state set
        if hasattr(adata_pp.uns,'neighbors') == False: #if neighbors haven't been computed, do so 
            scv.pp.neighbors(adata_pp, n_neighbors=k, n_pcs=n_pcs)

    sc.tl.leiden(adata_pp, resolution = resolution, random_state=0)

    clusters = adata_pp.obs['leiden'].copy()

    return clusters

def run_mnnCorrect(adata: AnnData,
                    X: list = None,
                    batches: list = None,
                    batch_correct_flavor: str = 'concat'):
    """performs batch effect correction using mutual nearest neighbors: https://www.nature.com/articles/nbt.4091
       implemented in R: https://rdrr.io/bioc/batchelor/man/mnnCorrect.html

    Parameters
    adata: AnnData
        Annotated data object
    X: list (default = None)
        list of spliced and unspliced normalized counts. Should be log transformed for mnn. 
            If provided: dimensions must be [(cells, genes), (cells, genes)].
            If not provided: X, unspliced layers will be used
    batches: list (default = None)
        list referring to batch condition of every cell. Dimensions must be (1 x cells) 
    batch_correct_flavor: str (default = 'concat')
        string referring to type of correction. Can be hansen or concat where hansen is sum and concat is concatenation
    ----------

    Returns
    adata: AnnData
        batch corrected annotated adata object.
            adata.X: log transformed corrected spliced counts
            adata.layers['spliced']: corrected spliced counts
            adata.layers['unspliced']: corrected unspliced counts

    ----------
    """
    if X is None:
        X = [adata.X.todense(), adata.layers['unspliced'].todense()]
    elif scipy.sparse.issparse(X[0]) or scipy.sparse.issparse(X[1]):
        X = [X[0].todense(), X[1].todense()]

    if batch_correct_flavor == 'hansen':
        sys.stdout.write('performing batch effect correction with mnn using hansen'+'\n')
        M, R = _to_hansen(X)
    elif batch_correct_flavor == 'concat':
        sys.stdout.write('performing batch effect correction with mnn using concatenation'+'\n')
        M = _to_concat(X)
    else:
        sys.stderr.write('batch effect correction flavor not recognized'+'\n')

    r = robjects.r
    r['source'](os.path.join('evi','preprocessing','preprocess.R'))
    run_mnnCorrect_r = robjects.globalenv['run_mnnCorrect']

    X_corr = run_mnnCorrect_r(X = pandas2ri.py2rpy(pd.DataFrame(M).transpose()),
                              batches = pandas2ri.py2rpy(pd.DataFrame(batches).transpose()))

    X_corr = X_corr[0].transpose() #put back genes x cells

    if batch_correct_flavor == 'hansen':
        X1_corr, X2_corr = _from_hansen(X_corr, M, R)
    elif batch_correct_flavor == 'concat':
        X1_corr, X2_corr = _from_concat(X_corr)

    adata.X = scipy.sparse.csr_matrix(X1_corr) #reset layers, this is corrected data which is in log space
    adata.layers['spliced'] = scipy.sparse.csr_matrix(np.expm1(X1_corr)) #spliced and unspliced matrices are scaled but not log transformed  
    adata.layers['unspliced'] = scipy.sparse.csr_matrix(np.expm1(X2_corr))
    
    return adata

def run_combat(adata: AnnData,
                X: list = None,
                batches: list = None,
                batch_correct_flavor: str = 'concat'):
    
    """performs batch effect correction on spliced and unspliced data using combat: https://academic.oup.com/biostatistics/article/8/1/118/252073,
                                                                                    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3307112/,
                                                                                    https://www.biorxiv.org/content/10.1101/2020.03.17.995431v2.full
    Parameters
    adata: AnnData
        Annotated data object
    X: list (default = None)
        list of spliced and unspliced normalized counts. Should be log transformed for mnn. 
            If provided: dimensions must be [(genes, cells), (genes, cells)].
            If not provided: X, unspliced layers will be used
    batches: list (default = None)
        list referring to batch condition of every cell. Dimensions must be (1 x cells) 
    batch_correct_flavor: str (default = None)
        string referring to type of correction. Can be hansen or concat where hansen is sum and concat is concatenation
    ----------

    Returns
    adata: AnnData
        batch corrected annotated adata object.
            adata.X: log transformed corrected spliced counts
            adata.layers['spliced']: corrected spliced counts
            adata.layers['unspliced']: corrected unspliced counts

    ----------
    """
    if X is None:
        X = [adata.X.todense(), adata.layers['unspliced'].todense()]
    elif scipy.sparse.issparse(X[0]) or scipy.sparse.issparse(X[1]):
        X = [X[0].todense(), X[1].todense()]

    if batch_correct_flavor == 'hansen':
        sys.stdout.write('performing batch effect correction with combat using hansen'+'\n')
        M, R = _to_hansen(X)
    elif batch_correct_flavor == 'concat':
        sys.stdout.write('performing batch effect correction with combat using concatentation'+'\n')
        M = _to_concat(X)
    else:
        sys.stderr.write('batch effect correction flavor not recognized'+'\n')

    adata_pp = AnnData(M)
    adata_pp.obs['batch'] = batches.copy()

    sc.pp.combat(adata_pp, key = 'batch')
    X_corr = adata_pp.X.copy()
                         
    if batch_correct_flavor == 'hansen':
        X1_corr, X2_corr = _from_hansen(X_corr, M, R)
    elif batch_correct_flavor == 'concat':
        X1_corr, X2_corr = _from_concat(X_corr)

    adata.X = scipy.sparse.csr_matrix(X1_corr) #reset layers, this is corrected data which is in log space
    adata.layers['spliced'] = scipy.sparse.csr_matrix(np.expm1(X1_corr)) #spliced and unspliced matrices are scaled but not log transformed so invert
    adata.layers['unspliced'] = scipy.sparse.csr_matrix(np.expm1(X2_corr))
    
    return adata

def run_scanorama(adata: AnnData,
                    X: list = None,
                    batches: list = None,
                    batch_correct_flavor: str = 'concat'):
    """performs batch effect correction on spliced and unspliced data using scanorama: https://www.nature.com/articles/s41587-019-0113-3

    Parameters
    adata: AnnData
        Annotated data object
    X: list (default = None)
        list of spliced and unspliced normalized counts. Should be log transformed for mnn. 
            If provided: dimensions must be [(genes, cells), (genes, cells)].
            If not provided: X, unspliced layers will be used
    batches: list (default = None)
        list referring to batch condition of every cell. Dimensions must be (1 x cells) 
    batch_correct_flavor: str (default = None)
        string referring to type of correction. Can be hansen or concat where hansen is sum and concat is concatenation
    ----------

    Returns
    adata: AnnData
        batch corrected annotated adata object.
            adata.X: log transformed corrected spliced counts
            adata.layers['spliced']: corrected spliced counts
            adata.layers['unspliced']: corrected unspliced counts

    ----------
    """
    import scanorama
    if X is None:
        X = [adata.X.todense(), adata.layers['unspliced'].todense()]
    elif scipy.sparse.issparse(X[0]) or scipy.sparse.issparse(X[1]):
        X = [X[0].todense(), X[1].todense()]

    if batch_correct_flavor == 'hansen':
        sys.stdout.write('performing batch effect correction with scanorama using hansen'+'\n')
        M, R = _to_hansen(X)
    elif batch_correct_flavor == 'concat':
        sys.stdout.write('performing batch effect correction with scanorama using concatentation'+'\n')
        M = _to_concat(X)
    else:
        sys.stderr.write('batch effect correction flavor not recognized'+'\n')

    adata_pp = AnnData(M)
    adata_pp.obs['batch'] = batches.copy()

    adatas = []

    for batch in np.unique(adata_pp.obs['batch']):
        batch_ind = np.where(adata_pp.obs['batch'] == batch)[0]
        adata_batch = adata_pp[batch_ind, :]
        adatas.append(adata_batch)

    scanorama_int = scanorama.correct_scanpy(adatas)
    scanorama_corr = [ad.X.todense() for ad in scanorama_int]

    # make into one matrix.
    X_corr = np.concatenate(scanorama_corr)
                         
    if batch_correct_flavor == 'hansen':
        X1_corr, X2_corr = _from_hansen(X_corr, M, R)
    elif batch_correct_flavor == 'concat':
        X1_corr, X2_corr = _from_concat(X_corr)

    adata.X = scipy.sparse.csr_matrix(X1_corr) #reset layers, this is corrected data which is in log space
    adata.layers['spliced'] = scipy.sparse.csr_matrix(np.expm1(X1_corr)) #spliced and unspliced matrices are scaled but not log transformed so invert
    adata.layers['unspliced'] = scipy.sparse.csr_matrix(np.expm1(X2_corr))
    
    return adata

def run_hvg(adata: AnnData, 
           flavor: str = 'seurat',
           min_disp: float = None,
           min_mean: float = None,
           max_mean: float = None,
           batch_key: str = None,
           n_top_genes: str  = None):
    """performs highly variable gene selection based upon suerat and dispersion criteria.

    Parameters
    adata: AnnData
        Annotated data object
    flavor: str (default = None)
        string referring to the method for identifying highly variable genes.
            if flavor = 'seurat' or = 'cellranger': expects log transformed data
            if flavor ='seurat_v3': expects non-log normalized data
    min_disp: float (default = None)
        float referring to the minimum cutoff for dispersion
    min_mean: float (default = None)
        float referring to the minumum mean cutoff
    max_mean: float (default = None)
        float referring to the maximum mean cutoff
    batch_key: str (default = None)
        string referring to the batch observations.
            if specified: highly variable genes are selected within each batch and merged
    n_top_genes: int (default = None)
        integer referring to the number of highly variable genes to keep
    ----------

    Returns
    adata: AnnData
        filtered annotated adata object with only highly variable genes returned
    ----------
    """
    sc.pp.highly_variable_genes(adata, flavor = flavor, min_disp = min_disp, min_mean = min_mean, max_mean = max_mean, batch_key = batch_key, n_top_genes = n_top_genes)
    
    # #taking all for prediction
    adata = adata[:, adata.var['highly_variable']]
    return adata

def preprocess(adata: AnnData,
                X: list = None,
                batches: list = None,
                mitostr: str = None,
                min_counts: int = None,
                max_counts: int = None,
                min_genes: int = None,
                max_genes: int = None,
                min_shared_counts: int = None,
                min_shared_cells: int = None,
                mito_ratio: float = None,
                normalize_method: str = None,
                batch_correct_method: str = None,
                batch_correct_flavor: str = None,
                clusters: list = None,
                n_pcs: int = 50,
                k: int = 10,
                resolution: float = 0.5,
                flavor: str = 'seurat',
                min_disp: float = None,
                min_mean: float = None,
                max_mean: float = None,
                batch_key: str = None,
                n_top_genes: str = None):
    """performs annotation of summary stats, cell and gene filtering, normalization,
        batch effect correction, highly variable gene selection, nearest neighbor estimation, and clustering.

    Parameters
    adata: AnnData
        Annotated data object
    mitostr: str
        string specifying mitochondarial genes. MT- for human, mt- for mouse
    X: list (default = None)
        list of spliced and unspliced counts. 
            If provided: dimensions must be [(genes, cells), (genes, cells)] for batchelor, scran, mnn, a.
            If not provided: X, unspliced layers will be used
    batches: list (default = None)
        list referring to batch condition of every cell. Dimensions must be (1 x cells) 
    min_counts: int (default = None)
        integer referring to the minimum number of gene expression counts (dropout)
    max_counts: int (default = None)
        integer referring to the maximum number of gene expression counts (doublet)
    min_genes: int (default = None)
        integer referring to the minimum number of genes required for a cell (low quality)
    max_genes: int (default = None)
        integer referring to the maximum number of genes required for a cell
    min_shared_counts: int (default = None)
        integer referring to the minimum number of unspliced and spliced counts required for a gene
    min_shared_cells: int (default = None)
        integer referring to the number of cells required to be expressed in both spliced and unspliced
    mito_ratio: float (default = None)
        float referring to the ratio of mitochondrial transcripts. keep > ratio.
    normalize_method: str (default = None)
        string referring to the normalization approach. Can be batchelor, scran, or cpm
    batch_correct_method: str (default = None)
        string referring to the batch effect correction approach. Can be None, mnn, combat
    batch_correct_flavor: str (default = None)
        string referring to type of correction. Can be None, hansen, concat
    clusters: list (default = None)
        list referring to cluster annotation for every cell. Dimensions must be (1 x cells) 
            if not provided, will be computed through leiden clustering and resolution parameter
    n_pcs: int (default = 50)
        integer referring to the number of principal components
            if == 0, use all data for nearest neighbors
    k: int (default = 10)
        integrer referring to the number of nearest neighbors
    resolution: float (default = 0.5)
        float referring to the resolution parameter for leiden clustering
    flavor: str (default = None)
        string referring to the method for identifying highly variable genes.
            if flavor = 'seurat' or = 'cellranger': expects log transformed data
            if flavor ='seurat_v3': expects non-log normalized data
    min_disp: float (default = None)
        float referring to the minimum cutoff for dispersion
    min_mean: float (default = None)
        float referring to the minumum mean cutoff
    max_mean: float (default = None)
        float referring to the maximum mean cutoff
    batch_key: str (default = None)
        string referring to the batch observations.
            if specified: highly variable genes are selected within each batch and merged
    n_top_genes: int (default = None)
        integer referring to the number of highly variable genes to keep
    ----------

    Returns
    adata: AnnData
        filtered, normalized and corrected annotated adata object
    ----------
    """
    if batches is not None:
        batch_key = 'batch'
        adata.obs['batch'] = batches.copy() #append so it's filtered when cells are
    else:
        batch_key = None
        
    adata = annotate_stats(adata = adata, mitostr = mitostr)
    adata = filter_cells_genes(adata = adata, min_counts = min_counts, max_counts = max_counts, min_genes = min_genes, max_genes = max_genes,
                                min_shared_counts = min_shared_counts, min_shared_cells = min_shared_cells, mito_ratio = mito_ratio)
    
    ##perform normalization
    if normalize_method == 'cpm':
        sys.stdout.write('performing cpm normalization'+'\n')  
        scv.pp.normalize_per_cell(adata, counts_per_cell_after = 1e6)
    elif normalize_method == 'batchelor':
        sys.stdout.write('estimating size factors with batchelor'+'\n')  
        adata = run_multiBatchNorm(adata, X = X, batches = adata.obs['batch'].values.tolist(), clusters = clusters, n_pcs = n_pcs, k = k, resolution = resolution)
    elif normalize_method == 'scran':
        sys.stdout.write('estimating sum factors with scran'+'\n')  
        adata = run_scran(adata, X = X, clusters = clusters, n_pcs = n_pcs, k = k, resolution = resolution)
    elif normalize_method is None:
        sys.stdout.write('skipping normalization'+'\n')   
    else:
        sys.stdout.write('normalization method not recognized'+'\n')
        
    ##find highly variable genes. This should really be performed post batch correction, but scanpy takes into account batch so to speed up runtime, we'll run this here. Also mnn tends to run this first
    sys.stdout.write('performing highly variable gene selection'+'\n')
    if flavor != 'seurat_v3':
        sc.pp.log1p(adata)
        adata = run_hvg(adata, flavor = flavor, min_disp = min_disp, min_mean = min_mean, max_mean = max_mean, batch_key = batch_key, n_top_genes = n_top_genes)
        adata.X = scipy.sparse.csr_matrix(np.expm1(adata.X.todense()))
        #^putting back on original scale for different batch effect correction methods that require this
    else:
        adata = run_hvg(adata, flavor = flavor, min_disp = min_disp, min_mean = min_mean, max_mean = max_mean, batch_key = batch_key)

    sys.stdout.write('dimensions following hvg: {}'.format(adata.shape)+'\n')

    #perform batch effect correction
    if batch_correct_method == 'mnn':
        adata = run_mnnCorrect(adata = adata, X = X, batches = adata.obs['batch'].values.tolist(), batch_correct_flavor = batch_correct_flavor) #this log transforms X
    elif batch_correct_method == 'combat':
        adata = run_combat(adata = adata, X = X, batches = adata.obs['batch'].values.tolist(),batch_correct_flavor = batch_correct_flavor) #this log transforms X
    elif batch_correct_method == 'scanorama':
        adata = run_scanorama(adata = adata, X = X, batches = adata.obs['batch'].values.tolist(),batch_correct_flavor = batch_correct_flavor) #this log transforms X
    else:
        sys.stdout.write('log transforming'+'\n')
        adata.X = scipy.sparse.csr_matrix.log1p(adata.X)
        sys.stdout.write('batch correction method not recognized'+'\n')

    ##estimate k-nearest neighbors and cluster the data
    if n_pcs == 0:
        sys.stdout.write('computing nearest neighbors from entire spliced data'+'\n') 
        scv.pp.neighbors(adata, n_neighbors=k, n_pcs=0)
    else:
        sys.stdout.write('performing PCA'+'\n')  
        sc.tl.pca(adata, n_comps = n_pcs, zero_center = True, svd_solver = 'arpack', random_state = 0) #random state set
        if hasattr(adata.uns,'neighbors') == False: #if neighbors haven't been computed, do so 
            sys.stdout.write('estimating k-nearest neighbors'+'\n')  
            scv.pp.neighbors(adata, n_neighbors=k, n_pcs=n_pcs)
    
    sys.stdout.write('performing leiden clustering'+'\n') 
    sc.tl.leiden(adata, resolution = resolution, random_state = 0)
    
    return adata

def plot_stats(adata: AnnData,
                groupby:str = None,
                n_counts_less: int = 500,
                n_counts_greater: int = 10000,
                n_genes_less: int = 200,
                mitostr: str = 'MT-'):
    """plots counts, genes, and mito ratio for quality control thresholding.

    Parameters
    adata: AnnData
        Annotated data object

    groupby: str (default = None),
        string referring to the groups in violin plot
    n_counts_less: int (default = 500)
        integer referring to the x range cutoff for gene counts for cells in the plot < # 
    n_counts_greater: int (default = 10000)
        integer referring to the x range cutoff for gene counts for cells in the plot > #
    n_genes_less: int (default = 200)
        integer referring to the number of genes for a cell cutoff in the plot
    mitostr: str (default = 'MT-')
        string referring to the mitochondrial gene identifier. 'MT-' for humans, 'mt-' for mouse
    ----------

    Returns
    ----------
    """
    adata = annotate_stats(adata = adata, mitostr = mitostr)
    adata.var_names_make_unique()
    t1 = sc.pl.violin(adata, 'n_counts', groupby=groupby, size=2, log=True, cut=0)
    t2 = sc.pl.violin(adata, 'mito_ratio', groupby=groupby)
    p1 = sc.pl.scatter(adata, 'n_counts', 'n_genes', color='mito_ratio')
    p2 = sc.pl.scatter(adata[adata.obs['n_counts']<n_counts_less], 'n_counts', 'n_genes', color='mito_ratio')
    p3 = sns.distplot(adata.obs['n_counts'], kde=False)
    plt.show()
    p4 = sns.distplot(adata.obs['n_counts'][adata.obs['n_counts']<n_counts_less], kde=False, bins=60)
    plt.show()
    p5 = sns.distplot(adata.obs['n_counts'][adata.obs['n_counts']>n_counts_greater], kde=False, bins=60)
    plt.show()
    p6 = sns.distplot(adata.obs['n_genes'], kde=False, bins=60)
    plt.show()
    p7 = sns.distplot(adata.obs['n_genes'][adata.obs['n_genes']<n_genes_less], kde=False, bins=60)
    plt.show()

def _to_hansen(X):
    """Computes matrices for performing batch effect correction of spliced and unspliced counts.
       idea for batch effect correcting spliced and unspliced counts credited to Hansen lab from thread: https://www.hansenlab.org/velocity_batch

       M = S + U
       R = S / (S + U)

       Sb = Mb*R
       Ub = Mb*(1-R)

    X: list
        list of matrices corresponding to data. Dimensions are 2*n x p
    ----------

    Returns
    M: ndarray
        matrix of summed spliced and unspliced counts. This is log transformed
    R: ndarray 
        matrix to transform data back to original 
    ----------
    """
    X1, X2 = evi.tl.check_sparse(X[0], X[1], return_sparse = False)
    
    M = X1 + X2
    R = X1 / (X1 + X2)
    sys.stdout.write('log transforming'+'\n')
    M = np.log1p(M)
    
    return M, R
    
    
def _from_hansen(X, M, R):
    """Transform corrected counts back to individual data matrices

    X: ndarray
        matrix referring to the batch corrected data
    M: ndarray
        matrix of summed spliced and unspliced counts. This is log transformed
    R: ndarray 
        matrix to transform data back to original 
    ----------

    Returns
    X1: ndarray
        matrix referring to the batch corrected data of first data type in log scale
    X2: ndarray 
        matrix referring to the batch corrected data of second data type in log scale
    ----------
    """
    X1_corr = np.multiply(X, R) 
    X2_corr = np.multiply(X, (1 - R))

    #nans induced with division, replace with 0s
    X1_corr = np.nan_to_num(X1_corr, 0)
    X2_corr = np.nan_to_num(X2_corr, 0)
    
    return X1_corr, X2_corr

def _to_concat(X):
    """horizionally concatenate spliced and unspliced counts. Then log transform
    
    X: list
        list of matrices corresponding to data. Dimensions are 2*n x p
    ----------

    Returns
    M: ndarray
        log transformed matrix of horizontally concatenated spliced and unspliced counts
    ----------
    """
    X = np.concatenate(X, axis = 1)
    sys.stdout.write('log transforming'+'\n')
    X = np.log1p(X)
    
    return X
    
    
def _from_concat(X):
    """Transform corrected counts back to individual data matrices

    X: ndarray
        matrix referring to the batch corrected data
    ----------

    Returns
    X1: ndarray
        matrix referring to the batch corrected data of first data type in log scale
    X2: ndarray 
        matrix referring to the batch corrected data of second data type in log scale
    ----------
    """
    X1 = X[:, :int(np.shape(X)[1]/2)] #cells x genes for X or spliced
    X2 = X[:, int(np.shape(X)[1]/2):] #cells x genes for unspliced
    
    return X1, X2
