import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid

class EVI(BaseEstimator):
    """Class for evaluating multi-modal data integration approaches for combining unspliced, spliced, and RNA velocity gene expression modalities

        Parameters
        ----------------------------
        adata: AnnData
            Annotated data object
        x1key: str (default = None)
            string referring to the layer of first matrix in the AnnData object. Can be X, Ms, spliced, unspliced, velocity, or None
        x2key: str (default = None)
            string referring to the layer of second matrix in the AnnData object. Can be X, Ms, spliced, unspliced, velocity, or None
        X1: (default = None)
            matrix referring to the first data type if x1key unspecified
        X2: (default = None)
            matrix referring to the second data type if x2key unspecified
        logX1: bool (default = None)
            boolean referring to whether the first data type should be log transformed. If data type is Ms or velocity, this should be False.
        logX2: bool (default = None)
            boolean referring to whether the second data type should be log transformed. If data type is Ms or velocity, this should be False.
        labels_key: str (default = None)
            string referring to the key in adata.obs of ground truth labels
        labels: (default = None)
            array referring to the labels for every cell
        int_method: function (default = None)
            function housed in the evi.tl.merge script that specifies the integration method to perform. Can be one of the following (or you may provide your own):
                evi.tl.expression
                evi.tl.moments
                evi.tl.concat_merge
                evi.tl.sum_merge
                evi.tl.cellrank
                evi.tl.snf
                evi.tl.precise
                evi.tl.precise_consensus
                evi.tl.grassmann
                evi.tl.integrated_diffusion
                evi.tl.mofap
                evi.tl.seurat_v4
        int_method_params: dictionary (default = None)
            dictionary referring to the integration method hyperparameters. For more information on method-specific hyperparameters, see the evi.tl.merge script for the method of interest. Can be:
                evi.tl.expression example: {'k': 10}
                evi.tl.moments example: {'k': 10}
                evi.tl.concat_merge example: {'k': 10}
                evi.tl.sum_merge example: {'k': 10}
                evi.tl.cellrank example: {'lam':0.7, 'scheme': 'correlation', 'mode':'deterministic'}
                evi.tl.snf example: {'k': 10, 'mu' : 0.5, 'K': 50}
                evi.tl.precise example: {'n_pvs': 30}
                evi.tl.precise_consensus example: {'n_pvs': 30}
                evi.tl.grassmann example: {'k': 10, 't' : 100, 'K': 50, 'lam': 1}
                evi.tl.integrated_diffusion example: {'k': 10, 'n_clusters' : 5, 'K': 50}
                evi.tl.mofap example: {'K': 50}
                evi.tl.seurat_v4 example: {'k': 10}
        eval_method: function (default = None)
            function housed in the evi.tl.infer script that specifies the evaluation method to perform. Can be one of the following (or you may provide your own):
                label propagation classification: evi.tl.lp
                support vector machine classification: evi.tl.svm
                trajectory inference evaluation: evi.tl.ti
        eval_method_params: dictionary (default = None)
            dictionary referring to the evaluation method hyperparameters. For more information on evaluation method -specific hyperparameters, see the evi.tl.infer script for the method of interest. Can be:
                evi.tl.lp example: {'train_size': 0.5, 'random_state': 0, 'metrics': ['F1', 'balanced_accuracy', 'auc', 'precision', 'accuracy']}
                evi.tl.svm example: {'random_state': 0, 'metrics': ['F1', 'balanced_accuracy', 'auc', 'precision', 'accuracy']}
                evi.tl.ti example: {'root_cluster': root_cluster, 'n_dcs': 20, 'connectivity_cutoff':0.05, 'root_cell': 646, 'ground_trajectory': ground_trajectory} or
                                   {'root_cluster': [root_cluster], 'n_dcs': [20], 'connectivity_cutoff':[0.05], 'root_cell':[646, 10, 389], 'ground_trajectory': [ground_trajectory]} 
        n_jobs: int (default = 1)
            number of jobs to use in computation 

        Attributes
        ----------------------------
        model.integrate()
            performs integration of gene expression modalities

            Returns:
                W: sparse graph adjacency matrix of combined data
                embed: embedding of combined data


        model.evaluate_integrate()
            performs integration of gene expression modalities and then evaluates method according to the evaluation criteria or task of interest

            Returns:
                score_df: dataframe of classification or trajectory inferences scores

        Examples
        ----------------------------
        1. Example for SVM classification using one data modality - spliced gene expression:

            model = evi.tl.EVI(adata = adata, x1key = 'spliced', logX1 = True,
                                labels_key = 'condition_broad', int_method = evi.tl.expression,
                                int_method_params = {'k': 10}, eval_method = evi.tl.svm,
                                eval_method_params = {'random_state': 0, 'metrics': ['F1', 'balanced_accuracy', 'auc', 'precision', 'accuracy']}, n_jobs = -1)

            W, embed = model.integrate()
            df = model.evaluate_integrate()

        2. Example for label propagation classification following spliced and unspliced integration using PRECISE:

            model = evi.tl.EVI(adata = adata, x1key = 'spliced', x2key = 'unspliced', logX1 = True, logX2 = True,
                                labels_key = 'condition_broad', int_method = evi.tl.precise,
                                int_method_params = {'n_pvs': 30}, eval_method = evi.tl.lp,
                                eval_method_params = {'train_size': 0.5, 'random_state': 0, 'metrics': ['F1', 'balanced_accuracy', 'auc', 'precision', 'accuracy']}, n_jobs = -1)

            W, embed = model.integrate()
            df = model.evaluate_integrate()
            
        3. Example for trajectory inference following integration of moments of spliced and RNA velocity data using SNF:

            eval_method_params = {'root_cluster': 'LTHSC_broad', 'n_dcs': 20, 'connectivity_cutoff':0.05, 'root_cell': 646}

            ground_trajectory = evi.tl.add_ground_trajectory('gt_nestorowa.h5ad') #add h5ad trajectory inference object

            eval_method_params['ground_trajectory'] = ground_trajectory #append trajectory object to evaluation method dictionary

            model = evi.tl.EVI(adata = adata, x1key = 'Ms', x2key = 'velocity',
                                logX1 = False, logX2 = False, labels_key = 'cell_types_broad_cleaned',
                                int_method = evi.tl.snf, int_method_params = {'k':10, 'mu':0.7, 'K': 50},
                                eval_method = evi.tl.ti, eval_method_params = eval_method_params, n_jobs = -1)

            df = model.evaluate_integrate()
        ----------
    """
    def __init__(
        self,
        adata=None,
        x1key=None,
        x2key=None,
        X1=None,
        X2=None,
        logX1=None,
        logX2=None,
        int_method=None,
        int_method_params=None,
        eval_method=None,
        eval_method_params=None,
        labels_key=None,
        labels=None,
        n_jobs=1,
        **int_kwargs
    ):
        
        self.adata = adata
        self.x1key = x1key
        self.x2key = x2key
        self.X1 = X1
        self.X2 = X2
        self.logX1 = logX1
        self.logX2 = logX2
        self.int_method = int_method
        self.int_method_params = int_method_params
        self.eval_method = eval_method
        self.eval_method_params = eval_method_params
        self.int_kwargs = int_method_params
        self.labels_key = labels_key
        self.labels = labels
        self.n_jobs = n_jobs
        
    def evaluate_integrate(self):
        sys.stdout.write('integration method: {}'.format(self.int_method.__name__)+'\n')
        self.score_df = pd.DataFrame()
        sys.stdout.write('performing integration with parameter set: {}'.format(self.int_method_params)+'\n')
        W, embedding = self.integrate()
        sys.stdout.write('evaluating: {}'.format(self.eval_method.__name__)+'\n')
        if W is not None:            
            if self.x2key is None:
                self.dtype = self.x1key
                self.dtype_total = self.x1key
            else:
                self.dtype = self.x1key + '_' + self.x2key
                self.dtype_total = self.x1key + '_' + self.x2key

            self.W = W
            self.embedding = embedding
            try: #if param sweep for evaluatation
                params_e = list(ParameterGrid(self.eval_method_params))
                for e in params_e:
                    self.eval_kwargs = e.copy()
                    self.scores, self.metric_labels = self._evaluate()
                    self.eval_kwargs.pop('ground_trajectory', None) #removes trajectory from string
                    self.eval_kwargs.pop('metrics', None) #removing metrics from param string
                    self._aggregate() #aggregate is necessary, combines ind with merged modalities 
            except:
                self.eval_kwargs = self.eval_method_params.copy()
                self.scores, self.metric_labels = self._evaluate()
                self.eval_kwargs.pop('ground_trajectory', None)
                self.eval_kwargs.pop('metrics', None)
                self._aggregate()

            if hasattr(self, 'scores'):
                self.score_df.index = self.metric_labels
                return self.score_df

    def integrate(self):
        W = self.int_method(adata = self.adata, x1key = self.x1key, x2key = self.x2key, X1 = self.X1, X2 = self.X2,
                            logX1 = self.logX1, logX2 = self.logX2, n_jobs = self.n_jobs, **self.int_kwargs)
        return W
    
    def _evaluate(self):
        scores, metric_labels = self.eval_method(adata = self.adata, W = self.W, embedding = self.embedding, labels_key = self.labels_key, labels = self.labels, n_jobs = self.n_jobs, **self.eval_kwargs)
        return scores, metric_labels
    
    def _aggregate(self):
        if len(self.int_kwargs) != 0:
            p1 = self._param(self.int_kwargs)
        else:
            p1 = 'default_int_params'
        if len(self.eval_kwargs)  != 0:
            p2 = self._param(self.eval_kwargs)
        else:
            p2 = 'default_eval_params'
        p = str(self.int_method.__name__) + '_' + self.dtype + '_' + p1 + '_'+ str(self.eval_method.__name__) + '_' + p2
        self.score_df = pd.concat([self.score_df.reset_index(drop=True), pd.DataFrame(self.scores, columns = [p])], axis = 1)

    def _param(self, kwargs):
        p = np.concatenate(list(map(list, kwargs.items())))
        p = '_'.join(p)
        return p