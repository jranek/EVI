import scvelo as scv  
from anndata import AnnData
import numpy as np

def velocity(adata: AnnData,
            n_pcs: int = 50,
            k: int = 30,
            var_names = 'all',
            mode: str = 'dynamical',
            mode_neighbors: str = 'distances',
            n_jobs: int = 8):
    """computes RNA velocity estimation with scvelo: https://scvelo.readthedocs.io/

    Parameters
    adata: AnnData
        Annotated data object
    n_pcs: int (default = 50)
        integer referring to the number of principal compoennts to use for spliced and unspliced moment computation
    k: int (default = 30)
        integer referring to the number of nearest neighbors used in moment computation and velocity graph computation
    var_names: str or list of str (default = 'all')
        string or list of strings referring to the genes to use for the fitting. 
            if 'all': all variable names will be used
            if 'velocity_genes': velocity genes will be used in the fitting
    mode: str (default = 'dynamical')
        string referring to which RNA velocity model implemented. Can be deterministic, stochastic, or dynamical
    mode_neighbors: str (default = 'distances')
        string referring to the type of KNN graph used. Can be distances or connectivities
    n_jobs: int (default = 8)
        integer referring to the number of parallel jobs
    ----------

    Returns
    adata: AnnData
        annotated adata object with RNA velocity information
        adata.layers['velocity']
        adata.uns['velocity_graph']
        adata.layers['Ms']
    ----------
    """
    if var_names == 'all':
        var_names = adata.var_names.values

    scv.pp.moments(adata, n_pcs = n_pcs, n_neighbors = k)
    scv.tl.recover_dynamics(adata, var_names = var_names, n_jobs = n_jobs)
    scv.tl.velocity(adata, mode = mode)
    scv.tl.velocity_graph(adata, n_neighbors = k, mode_neighbors = mode_neighbors)

    return adata

def velocity_corrected(adata: AnnData,
                        groupby: str = 'leiden',
                        k: int = 30,
                        var_names: str = 'all',
                        mode: str = 'dynamical',
                        mode_neighbors: str = 'distances',
                        likelihood_threshold: float = 0.001):
    """corrects RNA velocity estimation with scvelo according to differential kinetics: https://scvelo.readthedocs.io/

    Parameters
    adata: AnnData
        Annotated data object
    groupby: str (default = 'leiden')
        string referring to the obs value of clusters. This will be used to perform differential kinetics test on
    k: int (default = 30)
        integer referring to the number of nearest neighbors used in moment computation and velocity graph computation
    var_names: str or list of str (default = 'all')
        string or list of strings referring to the genes to use for the fitting. 
            if 'all': all variable names will be used
            if 'velocity_genes': velocity genes will be used in the fitting
    mode: str (default = 'dynamical')
        string referring to which RNA velocity model implemented. Can be deterministic, stochastic, or dynamical
    mode_neighbors: str (default = 'distances')
        string referring to the type of KNN graph used. Can be distances or connectivities
    likelihood_threshold: float (default = 0.001)
        float referring to the likelihood threshold of genes to keep
    ----------

    Returns
    adata: AnnData
        annotated adata object with correcred velocity information
        adata.layers['velocity']
        adata.uns['velocity_graph']
        adata.layers['Ms']
    ----------
    """
    if var_names == 'all':
        var_names = adata.var_names.values

    likelihood_index = np.where(adata.var['fit_likelihood'].sort_values(ascending = False).values >= likelihood_threshold)[0]
    top_genes = adata.var['fit_likelihood'].sort_values(ascending = False).index[likelihood_index].values

    scv.tl.differential_kinetic_test(adata, var_names = top_genes, groupby = groupby)
    scv.tl.velocity(adata, mode = mode, diff_kinetics = True)
    scv.tl.velocity_graph(adata, n_neighbors = k, mode_neighbors = mode_neighbors)

    return adata

def perform_velocity_estimation(adata: AnnData,
                                n_pcs: int = 50,
                                k: int = 30,
                                var_names: str = 'all',
                                mode: str = 'dynamical',
                                mode_neighbors: str = 'distances',
                                n_jobs: int = 8,
                                groupby: str = 'leiden',
                                likelihood_threshold: float = 0.001):
    """computes RNA velocity estimation and differential kinetics correction with scvelo: https://scvelo.readthedocs.io/

    Parameters
    adata: AnnData
        Annotated data object
    n_pcs: int (default = 50)
        integer referring to the number of principal compoennts to use for spliced and unspliced moment computation
    k: int (default = 30)
        integer referring to the number of nearest neighbors used in moment computation and velocity graph computation
    var_names: str or list of str (default = 'all')
        string or list of strings referring to the genes to use for the fitting. 
            if 'all': all variable names will be used
            if 'velocity_genes': velocity genes will be used in the fitting
    mode: str (default = 'dynamical')
        string referring to which RNA velocity model implemented. Can be deterministic, stochastic, or dynamical
    mode_neighbors: str (default = 'distances')
        string referring to the type of KNN graph used. Can be distances or connectivities
    n_jobs: int (default = 8)
        integer referring to the number of parallel jobs
    groupby: str (default = 'leiden')
        string referring to the obs value of clusters. This will be used to perform differential kinetics test on
    likelihood_threshold: float (default = 0.001)
        float referring to the likelihood threshold of genes to keep
    ----------

    Returns
    adata: AnnData
        annotated adata object with corrected RNA velocity information
        adata.layers['velocity']
        adata.uns['velocity_graph']
        adata.layers['Ms']
    ----------
    """
    adata = velocity(adata, n_pcs = n_pcs, k = k, var_names = var_names, mode = mode, mode_neighbors = mode_neighbors, n_jobs = n_jobs)
    adata = velocity_corrected(adata, groupby = groupby, k = k, var_names = var_names, mode = mode, mode_neighbors = mode_neighbors, likelihood_threshold = likelihood_threshold)
    return adata