import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid

class EVI(BaseEstimator):  
    def __init__(
        self,
        adata,
        x1key,
        x2key,
        X1,
        X2,
        logX1,
        logX2,
        int_method,
        int_method_params,
        eval_method,
        eval_method_params,
        labels_key,
        labels,
        n_jobs,
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

            self.W = W.copy()
            self.embedding = embedding.copy()
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
        p1 = self._param(self.int_kwargs)
        p2 = self._param(self.eval_kwargs)
        p = str(self.int_method.__name__) + '_' + self.dtype + '_' + p1 + '_'+ str(self.eval_method.__name__) + '_' + p2
        self.score_df = pd.concat([self.score_df.reset_index(drop=True), pd.DataFrame(self.scores, columns = [p])], axis = 1)

    def _param(self, kwargs):
        p = np.concatenate(list(map(list, kwargs.items())))
        p = '_'.join(p)
        return p