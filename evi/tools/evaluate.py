import evi
import pandas as pd
import numpy as np
import scipy
import harmonypy as hm
from sklearn.preprocessing import MinMaxScaler

def compute_lisi(adata, basis, batch_key, perplexity):
    X = adata.obsm[basis]
    metadata = pd.DataFrame(adata.obs[batch_key].values, columns = [batch_key])
    lisi = hm.compute_lisi(X, metadata, [batch_key], perplexity)
    return lisi

def corr_dist(adata_batch, adata, batch_label, batch_key):
    
    spliced_b = pd.DataFrame(adata_batch.layers['spliced'].todense(), index = adata_batch.obs_names, columns = adata_batch.var_names)
    unspliced_b = pd.DataFrame(adata_batch.layers['unspliced'].todense(), index = adata_batch.obs_names, columns = adata_batch.var_names)
    
    spliced_i = pd.DataFrame(adata.layers['spliced'].todense(), index = adata.obs_names, columns = adata.var_names)
    unspliced_i = pd.DataFrame(adata.layers['unspliced'].todense(), index = adata.obs_names, columns = adata.var_names)

    b = np.where(adata_batch.obs[batch_key] == batch_label)[0]
    
    corr_list = []
    for i in range(0, len(adata_batch.var_names)):
        df_b = pd.concat([spliced_b.iloc[b, i], unspliced_b.iloc[b, i]], axis = 1)
        cellind = df_b.iloc[np.where(df_b.sum(axis = 1) != 0)[0], :].index
        df_b = df_b.loc[cellind]
        mat_b = np.array(df_b.values)
        df_i = pd.concat([spliced_i.iloc[:, i], unspliced_i.iloc[:, i]], axis = 1)
        df_i = df_i.loc[cellind]
        mat_i = np.array(df_i.values)
        rho, pval = scipy.stats.spearmanr(scipy.spatial.distance.pdist(mat_b), scipy.spatial.distance.pdist(mat_i))
        corr_list.append(rho)
        
    return corr_list

def average_dataset_metric(df = None, m_order = None, metric = None, palette = None, figsize = None, save = False, filename = None):

    #computes ranked aggregate scores by min-max scaling, then taking the mean across datasets
    
    m = df[np.isin(df.index, m_order)]
    
    scaler = MinMaxScaler()
    m_ranked = pd.DataFrame(scaler.fit_transform(m), index = m.index, columns = m.columns)
    m_ranked = m_ranked.reindex(m_order)
    
    mean_metrics = pd.DataFrame(m_ranked.mean(1), columns = [metric])
    
    nplots = len(m_ranked.columns)

    evi.pl.ranked_barplot(df = m_ranked, figsize = figsize, y = m_ranked.index, save = save, palette = palette, filename = filename, nplots = nplots)
    
    return mean_metrics