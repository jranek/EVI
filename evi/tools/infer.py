import evi
import numpy as np
import pandas as pd 
import scanpy as sc
import scipy
from anndata import AnnData
from abc import abstractmethod
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, balanced_accuracy_score, confusion_matrix
from sklearn.svm import SVC
import os
#os.environ["R_HOME"] = r"C:\Users\Jolene\miniconda3\envs\venv_EVI\Lib\R"
#os.environ["PATH"]   = r"C:\Users\Jolene\miniconda3\envs\venv_EVI\Lib\R\bin\x64" + ";" + os.environ["PATH"]
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import rpy2.rinterface_lib.callbacks
import anndata2ri

pandas2ri.activate()
anndata2ri.activate()

class BaseLabelPropagation:
    """Class for performing label propagation

    Parameters
    W: ndarray
        adjacency matrix to compute label propagation on
    ----------

    Returns
    ----------
    """
    def __init__(self, W):
        self.W_norm = self._normalize(W)
        self.n_nodes = np.shape(W)[0]
        self.indicator_labels = None
        self.n_classes = None
        self.labeled_mask = None
        self.predictions = None

    @staticmethod
    @abstractmethod
    def _normalize(W):
        raise NotImplementedError("_normalize must be implemented")

    @abstractmethod
    def _propagate(self):
        raise NotImplementedError("_propagate must be implemented")


    def _encode(self, labels):
        # Get the number of classes
        classes = np.unique(labels)
        classes = classes[classes != -1] #-1 are unlabeled nodes so we'll exclude them
        self.n_classes = np.shape(classes)[0]
        # One-hot encode labeled data instances and zero rows corresponding to unlabeled instances
        unlabeled_mask = (labels == -1)
        labels = labels.copy()
        labels[unlabeled_mask] = 0
        onehot_encoder = OneHotEncoder(sparse = False)
        self.indicator_labels = labels.reshape(len(labels), 1)
        self.indicator_labels = onehot_encoder.fit_transform(self.indicator_labels)
        self.indicator_labels[unlabeled_mask, 0] = 0

        self.labeled_mask = ~unlabeled_mask

    def fit(self, labels, max_iter, tol):
        """Fits semisupervised label propagation model

        Parameters
        labels: ndarray
            labels for every node, where -1 indicates unlabeled nodes
        max_iter: int (default = 10000)
            maximum number of iterations before stopping prediction
        tol: float (default = 1e-3)
            float referring to the error tolerance between runs. If unchanging, stop prediction
        """
        self._encode(labels)

        self.predictions = self.indicator_labels.copy()
        prev_predictions = np.zeros((self.n_nodes, self.n_classes), dtype = np.float)

        for i in range(max_iter):
            # Stop iterations if the system is considered at a steady state
            variation = np.abs(self.predictions - prev_predictions).sum().item()

            if variation < tol:
                print(f"The method stopped after {i} iterations, variation={variation:.4f}.")
                break

            prev_predictions = self.predictions
            self._propagate()

    def predict(self):
        return self.predictions

    def predict_labels(self):
        """
        Returns
        predicted_labels: ndarray
            array of predicted labels according to the maximum probability
        predicted_scores: ndarray
            array of probability scores with dimensions n x nclasses
        uncertainty: ndarray
            array 1 - max of predictions

        ----------
        """
        predicted_labels = np.argmax(self.predictions, axis = 1)
        predicted_scores = self.predictions
        uncertainty = 1 - np.max(predicted_scores, 1)

        return predicted_labels, predicted_scores, uncertainty

class LabelPropagation(BaseLabelPropagation):
    def __init__(self, W):
        super().__init__(W)

    @staticmethod
    def _normalize(W):
        """ Computes row normalized adjacency matrix: D^-1 * W"""
        d = W.sum(axis=0).getA1()
        d = 1/d
        D = scipy.sparse.diags(d)

        return D @ W

    def _propagate(self):
        self.predictions = self.W_norm @ self.predictions

        # Put back already known labels
        self.predictions[self.labeled_mask] = self.indicator_labels[self.labeled_mask]

    def fit(self, labels, max_iter = 10000, tol = 1e-3):
        super().fit(labels, max_iter, tol)

def lp(adata: AnnData,
        W = None,
        embedding = None,
        labels_key: str = None,
        labels: list = None,
        train_size: float = 0.5,
        random_state: int = 0,
        metrics: list = ['F1', 'balanced_accuracy', 'auc', 'precision', 'accuracy'],
        **args):
    """performs label propagation for prediction.

    Parameters
    adata: AnnData
        Annotated data object
    W: (default = None)
        adjacency matrix to perform label propagation on
    embedding: ndarray (default = None)
        lower dimensional embedding to append to adata object
    labels_key: str (default = None')
        string referring to the ground truth labels key in adata object
    labels: list (default = None)
        membership values for all cells
    train_size: float (default = 0.5)
        percentage of labeled nodes to train on
    random_state: int (default = 0)
        for reproducibility
    metrics: list (default = ['F1', 'balanced_accuracy', 'auc', 'precision', 'accuracy'])
        type of metrics to compute for prediction evaluation
    ----------

    Returns
    scores: list
        prediction scores based upon the metrics specified
    metric_labels: list
        list of metric labels. Dimensions nsplits x 1
    ----------
    """
    adata_pred = evi.tl.convert(adata = adata, xkey = 'X', W = W, embedding = embedding, cluster_key = labels_key, clusters = labels)

    le = LabelEncoder()
    y = le.fit_transform(adata_pred.obs['labels']).astype(int)
    X = adata_pred.X.copy()
    n_nodes = np.shape(y)[0]
    n_classes = len(np.unique(adata_pred.obs['labels']))
    y_bin = label_binarize(y, classes=list(np.arange(0, n_classes)))
    n_splits = 10

    scores = []
    sss = StratifiedShuffleSplit(n_splits = n_splits, train_size = train_size, random_state = random_state)
    sss.get_n_splits(X, y = y)
    for train_index, test_index in sss.split(X, y):
        y_t = np.full(n_nodes, -1.)
        y_t[train_index] = y[train_index].copy()
        label_propagation = evi.tl.LabelPropagation(W)
        label_propagation.fit(y_t)
        predicted_labels, predicted_scores, _ = label_propagation.predict_labels()
        
        metric_values = []
        for m in metrics:
            if m == 'F1':
                metric_values.append(f1_score(y_true = y[test_index], y_pred = predicted_labels[test_index], average = 'weighted'))
            elif m == 'balanced_accuracy':
                metric_values.append(balanced_accuracy_score(y_true = y[test_index], y_pred = predicted_labels[test_index]))
            elif m == 'auc':
                if n_classes == 2: #binary 
                    #prob should be of the class with the greater label ie 1
                    metric_values.append(roc_auc_score(y_bin[test_index], predicted_scores[test_index, 1], multi_class = 'ovr'))
                else: #multiclass
                    metric_values.append(roc_auc_score(y_bin[test_index, :], predicted_scores[test_index, :], multi_class = 'ovr'))
            elif m =='precision':
                metric_values.append(precision_score(y_true = y[test_index], y_pred = predicted_labels[test_index], average = 'weighted'))
            elif m =='accuracy':
                metric_values.append(accuracy_score(y_true = y[test_index], y_pred = predicted_labels[test_index]))

        scores.append(metric_values)

    scores = np.asarray(scores).transpose().ravel() #F1... nsplit ... auc .... nsplit etc
    metric_labels = np.repeat(metrics, n_splits)
    return scores, metric_labels

def svm(adata: AnnData,
        W = None,
        embedding = None,
        labels_key: str = None,
        labels: list = None,
        random_state: int = 0,
        metrics: list = ['F1', 'balanced_accuracy', 'auc', 'precision', 'accuracy'],
        **args):
    """performs svm for prediction.

    Parameters
    adata: AnnData
        Annotated data object
    W: (default = None)
        adjacency matrix representing the graph
    embedding: ndarray (default = None)
        lower dimensional embedding to perform svm on
    labels_key: str (default = None')
        string referring to the ground truth labels key in adata object
    labels: list (default = None)
        membership values for all cells
    random_state: int (default = 0)
        for reproducibility
    metrics: list (default = ['F1', 'balanced_accuracy', 'auc', 'precision', 'accuracy'])
        type of metrics to compute for prediction evaluation
    ----------

    Returns
    scores: list
        prediction scores based upon the metrics specified
    metric_labels: list
        list of metric labels. Dimensions nsplits x 1
    ----------
    """
    adata_pred = evi.tl.convert(adata = adata, xkey = 'X', W = W, embedding = embedding, cluster_key = labels_key, clusters = labels)

    le = LabelEncoder()
    y = le.fit_transform(adata_pred.obs['labels']).astype(int)
    X = adata_pred.obsm['embedding'].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_classes = len(np.unique(adata_pred.obs['labels']))
    y_bin = label_binarize(y, classes=list(np.arange(0, n_classes)))
    n_splits = 10

    scores = []
    cv = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = random_state)
    for train_ix, test_ix in cv.split(X_scaled, y):
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        cv_hyperparam = StratifiedKFold(n_splits = 3, shuffle = True, random_state = random_state)
        model = SVC(random_state = 0, probability = True, max_iter= 10000)
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear']}
        gsearch = GridSearchCV(model, param_grid, scoring = 'balanced_accuracy', n_jobs = -1, cv = cv_hyperparam, refit = True)
        result = gsearch.fit(X_train, y_train)
        best_model = result.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)
        metric_values = []
        for m in metrics:
            if m == 'F1':
                metric_values.append(f1_score(y_true = y_test, y_pred = y_pred, average = 'weighted'))
            elif m == 'balanced_accuracy':
                metric_values.append(balanced_accuracy_score(y_true = y_test, y_pred = y_pred))
            elif m =='precision':
                metric_values.append(precision_score(y_true = y_test, y_pred = y_pred, average = 'weighted'))
            elif m =='accuracy':
                metric_values.append(accuracy_score(y_true = y_test, y_pred = y_pred))
            elif m == 'auc':
                if n_classes == 2: #binary 
                    #prob should be of the class with the greater label ie 1
                    metric_values.append(roc_auc_score(y_bin[test_ix], y_prob[:, 1], multi_class = 'ovr')) #y_prob is already on test
                else: #multiclass
                    metric_values.append(roc_auc_score(y_bin[test_ix, :], y_prob, multi_class = 'ovr'))
                
        scores.append(metric_values)
        
    scores = np.asarray(scores).transpose().ravel() #F1... nsplit ... auc .... nsplit etc
    metric_labels = np.repeat(metrics, n_splits)

    return scores, metric_labels

def construct_ground_trajectory(adata: AnnData,
                                cluster_key: str = None,
                                milestone_network = None,
                                counts = None,
                                expression = None,
                                group_ids = None,
                                cell_ids = None,
                                feature_ids = None,
                                directory: str = None,
                                data_ident: str = None,
                                adata_ident: str = None,
                                filename: str = None):
    """constructs a ground truth reference trajectory for comparison.

    Parameters
    adata: AnnData
        Annotated data object
    cluster_key: str (default = None)
        String referring to the ground truth labels key in adata object
    milestone_network:
        Trajectory network consisting of groups and edges between them
    counts:
        Raw count matrix of ground truth data
    expression:
        Normalized and log transformed matrix of ground truth data
    group_ids:
        Series object of cluster names for every cell
    cell_ids:
        Series object of cell names 
    feature_ids:
        Series object of feature names
    directory: str (default = None)
        String referring to the directory to save ground truth trajectory to
    data_ident: str (defult = None)
        String referring to the data name for saving. ex - Schafflick
    adata_ident: str (default = None)
        String referring to the adata object name for saving. ex - adata_Schafflick 
    filename: str (default = None)
        String referring to the filename for saving
    ----------

    Returns
    ----------
    """
    import dynclipy
    if cell_ids is None:
        cell_ids = adata.obs.index.copy()
    if feature_ids is None:
        feature_ids = list(adata.var_names)
    if group_ids is None:
        group_ids = list(adata.obs[cluster_key].values)
    if counts is None:
        counts = adata.layers['raw_spliced'].copy()
    if expression is None:
        expression = adata.X.copy()
        
    counts, expression = evi.tl.check_sparse(counts, expression, return_sparse = False)
    
    counts = pd.DataFrame(counts, index = cell_ids, columns = feature_ids)
    expression = pd.DataFrame(expression, index = cell_ids, columns = feature_ids)
    
    grouping = pd.DataFrame({'cell_id': cell_ids, 'group_id': group_ids})
    group_ids = list(np.unique(grouping['group_id']))
    milestone_ids = list(np.unique(np.concatenate(milestone_network.iloc[:, :2].values)))
    milestone_percentages = pd.DataFrame({'cell_id': grouping['cell_id'],
                                      'milestone_id': grouping['group_id'],
                                      'percentage': np.ones(shape = (1, len(grouping['cell_id']))).flatten()})

    ground_trajectory = dynclipy.wrap_data(cell_ids = cell_ids,
                                            feature_ids = feature_ids,
                                            grouping = grouping,
                                            group_ids = group_ids,
                                            milestone_network = milestone_network,
                                            milestone_ids = milestone_ids,
                                            milestone_percentages = milestone_percentages,
                                            counts = counts,
                                            expression = expression)
    
    evi.pp.make_directory(os.path.join(directory, 'ti', data_ident, adata_ident))
    ground_trajectory.write_output(file = filename+'.h5ad')
    os.replace(filename+'.h5ad', os.path.join(directory,'ti',data_ident, adata_ident,filename+'.h5ad'))
    
def add_ground_trajectory(directory: str = None,
                          filename: str = None):
    """adds ground truth trajectory into memory.
    directory: str (default = None)
        String referring to the directory to save ground truth trajectory to
    filename: str (default = None)
        String referring to the filename for saving
    ----------

    Returns
    trajectory:
        dynverse trajectory object
    ----------
    """
    r = robjects.r
    r['source'](os.path.join('evi', 'tools', 'infer.R'))
    add_ground_trajectory_r = robjects.globalenv['add_ground_trajectory']

    trajectory = add_ground_trajectory_r(directory = rpy2.robjects.vectors.StrVector([directory]),
                                     filename = rpy2.robjects.vectors.StrVector([filename]))
    return trajectory

def run_paga(adata: AnnData,
            cluster_key: str = None,
            root_cluster: str = None,
            root_cell: int = None,
            n_dcs: int = 10,
            connectivity_cutoff: float = 0.0001,
            model = 'v1.0',
            **args):
    """Performs PAGA + DPT pseudotime. Code pulled from dynverse container: https://github.com/dynverse/ti_paga/blob/master/run.py

    Parameters
    adata: AnnData
        Annotated data object
    cluster_key: str (default = None)
        string referring to the observation key of cluster annotations. This is used as input to PAGA
    root_cluster: str (default = None)
        cluster where root cell exists. If root cell is specified, this is ignored. If root cell unspecified, random cell is chosen from root cluster
    n_dcs: int (default = 10)
        number of diffusion components to use in DPT computation
    connectivity_cutoff: float (default = 0.0001)
        threshold of connections in the branch trajectory. Larger number indicates removing more connections
    model:
        version of PAGA to implement. Can be either v1.2 or v1.0. v1.0 indicates using the connectivities matrix
    ----------

    Returns
    adata: AnnData
        Annotated data object with DPT and PAGA information
    grouping: pd.DataFrame
        dataframe containing cells and cluster information
    branch_progressions: pd.DataFrame
        dataframe containing cells, branch connections, and percentage of cells along those branches
    branches: pd.DataFrame
        dataframe containing branch id, length computed frm max-min pseudotime, and direction information
    branch_network: pd.DataFrame
        dataframe containing the ordering of branches
    ----------
    """
    sc.tl.paga(adata, groups = cluster_key, model = model)
    
    if root_cell is None:
        root_cells = np.where(adata.obs[cluster_key] == root_cluster)[0]
        root_cell = np.random.choice(root_cells)
    
    adata.uns['iroot'] = root_cell
    
    sc.tl.diffmap(adata, n_comps = n_dcs)
    sc.tl.dpt(adata, n_dcs = n_dcs)

    grouping = pd.DataFrame({'cell_id': adata.obs.index, 'group_id': adata.obs[cluster_key]})

    # milestone network
    milestone_network = pd.DataFrame(np.triu(adata.uns['paga']['connectivities'].todense(), k = 0),
                                      index = adata.obs[cluster_key].cat.categories,
                                      columns = adata.obs[cluster_key].cat.categories).stack().reset_index()
    milestone_network.columns = ['from', 'to', 'length']
    milestone_network = milestone_network.query('length >= ' + str(connectivity_cutoff)).reset_index(drop = True)
    milestone_network['directed'] = False

    # branch progressions: the scaled dpt_pseudotime within every cluster
    branch_progressions = adata.obs.copy()
    branch_progressions['dpt_pseudotime'] = branch_progressions['dpt_pseudotime'].replace([np.inf, -np.inf], 1) # replace unreachable pseudotime with maximal pseudotime
    branch_progressions['percentage'] = branch_progressions.groupby(cluster_key)['dpt_pseudotime'].apply(lambda x: (x - x.min())/(x.max() - x.min())).fillna(0.5)
    branch_progressions['cell_id'] = adata.obs.index.copy()
    branch_progressions['branch_id'] = branch_progressions[cluster_key].astype(np.str)
    branch_progressions = branch_progressions[['cell_id', 'branch_id', 'percentage']]

    # branches:
    # - length = difference between max and min dpt_pseudotime within every cluster
    # - directed = not yet correctly inferred
    branches = adata.obs.groupby(cluster_key).apply(lambda x: x["dpt_pseudotime"].max() - x["dpt_pseudotime"].min()).reset_index()
    branches.columns = ['branch_id', 'length']
    branches['branch_id'] = branches['branch_id'].astype(np.str)
    branches['directed'] = True

    # branch network: determine order of from and to based on difference in average pseudotime
    branch_network = milestone_network[['from', 'to']]
    average_pseudotime = adata.obs.groupby(cluster_key)['dpt_pseudotime'].mean()
    for i, (branch_from, branch_to) in enumerate(zip(branch_network['from'], branch_network['to'])):
        if average_pseudotime[branch_from] > average_pseudotime[branch_to]:
            branch_network.at[i, 'to'] = branch_from
            branch_network.at[i, 'from'] = branch_to
            
    return adata, grouping, branch_progressions, branches, branch_network

def ti(adata: AnnData,
        W = None,
        ground_trajectory = None,
        labels_key: str = None,
        labels: list = None,
        root_cluster: str = None,
        root_cell: int = None,
        n_dcs: int = None,
        connectivity_cutoff: float = 0.05,
        model: str = 'v1.0',
        **args):

    """performs trajectory inference with PAGA and dpt to obtain a predicted trajectory. 
        predicted and reference trajectories are then evalauted with dynverse framework

    Parameters
    adata: AnnData
        Annotated data object
    W: 
        Matrix referring to the connectivity graph
    ground_trajectory:
        Dynverse trajectory object referring to ground truth reference
    labels_key: str (default = None)
        String referring to adata object obs value with cluster information
    labels: list (default = None)
        List of clusters annotations for every cell
    root_cluster: str (default = None)
        String with root cluster name to obtain a root cell from
    root_cell: int (default = None)
        Root cell index if available
    n_dcs: int (default = None)
        Number of diffusion map components for DPT
    connectivity_cutoff: float (default = None)
        Float referring to the connectivity cutoff for PAGA graph. This removes spurious connections
    model: str (default = 'v1.0')
        PAGA graph model
    ----------

    Returns
    scores: 
        score values following evaluation 
    metric_labels:
        labels of metrics used following evaluation 
    ----------
    """
    adata_pred = evi.tl.convert(adata = adata, xkey = 'X', W = W, embedding = None, cluster_key = labels_key, clusters = labels)

    # if type(n_dcs) is 'list':
    #     n_dcs = n_dcs[0]
    # if type(connectivity_cutoff) is 'list':
    #     connectivity_cutoff = connectivity_cutoff[0]

    adata_pred, grouping, branch_progressions, branches, branch_network = evi.tl.run_paga(adata_pred,
                                                                                            cluster_key = 'labels',
                                                                                            root_cluster = root_cluster,
                                                                                            root_cell = root_cell,
                                                                                            n_dcs = n_dcs,
                                                                                            connectivity_cutoff = connectivity_cutoff,
                                                                                            model = model)

    r = robjects.r
    r['source'](os.path.join('evi', 'tools', 'infer.R'))
    perform_evaluation_PAGA_r = robjects.globalenv['perform_evaluation_PAGA']

    result_r = perform_evaluation_PAGA_r(pandas2ri.py2rpy(grouping),
                                       pandas2ri.py2rpy(branch_progressions),
                                       pandas2ri.py2rpy(branches),
                                       pandas2ri.py2rpy(branch_network),
                                       ground_trajectory)

    scores = result_r[0]
    metric_labels = result_r[1]

    return scores, metric_labels