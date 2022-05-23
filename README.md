# EVI
Expression and Velocity Integration

## Introduction
EVI is a python package designed for evaluating multi-modal data integration strategies on combining unspliced, spliced, and RNA velocity gene expression modalities for trajectory inference and disease prediction tasks. For more details on the methods considered and the overall benchmarking pipeline, please read the associated preprint:

[Ranek, J. S., Stanley, N., and Purvis, J. E. Integrating temporal single-cell gene expression modalities for trajectory inference and disease prediction. _bioRxiv_, March 2022.](https://doi.org/10.1101/2022.03.01.482381)

## Overview
<p align="center">
  <img src="/doc/pipeline.png"/>
</p>

## Installation

* You can clone the git repository by, 

```
git clone https://github.com/jranek/EVI.git
```

* Once you've clone the repository, please change your working directory into this folder.

```
cd EVI
```

## Dependencies

Given that there are a number of required python and R packages for benchmarking evaluation, we recommend that you create a conda environment as follows. 

* First, create the conda environment using the yml file. This contains most of the installation instructions.

```
conda env create -f venv_EVI.yml
```

* Once the environment is created, you can activate it by,

```
conda activate venv_EVI
```

* Unfortunately, a few python and R packages are unavailable through the standard channels.
* To install the remaining dependencies, we provide a makefile which contains the python and R installation instructions. This must be done within the environment so make sure you have already activated it. 
* Of note, this step installs `dynverse` which is a collection of R packages required for trajectory inference evaluation. It uses the GitHub API which is limited to 60 requests; therefore, you may run into an `API rate limit exceeded` installation error. To increase the rate limit and overcome this, you can follow their guidelines here: https://dynverse.org/users/1-installation/. Namely, 
    * Open R and generate a GitHub token with `usethis::create_github_token()`. Alternatively, you can generate the token on GitHub by navigating to https://github.com/settings/tokens/new
    * Copy the GitHub token and set your personal access token within R using the `usethis::edit_r_environ()` command. This will open a tab for which you can set a GITHUB_PAT variable with your token by typing `GITHUB_PAT = 'your_github_token'` 
    * Save the file and close R
* To finish installation with the makefile, run

```
make
```

## Data access
You can download all of the preprocessed loom and h5ad files from the [Zenodo](https://zenodo.org/record/6110279#.Yg1jPN_MK3C) repository. This can be done directly from the website or through the terminal. 

If you'd like to list all of the publicly available files for download,

```python
from lxml import html
import requests

r = requests.get(f'https://zenodo.org/record/6110279#.Yg1jPN_MK3C')
content = html.fromstring(r.content)
hrefs = content.xpath('//a/@href')
files = [i for i in hrefs if i.endswith('?download=1')]
files = np.unique(files)
print(files)
```

If you'd like to download an example dataset from the command line, please specify both the zenodo url https://zenodo.org/ and the dataset string identifier from above.  

```
curl 'https://zenodo.org/record/6110279/files/adata_lane.h5ad?download=1' --output adata_lane.h5ad
```
## Tutorial
Given unspliced, spliced, or RNA velocity gene expression modalities, you can compare multi-modal data integration strategies using the evi class as follows. 

This class provides two methods following instantiation, `integrate()` which can be used to integrate gene expression modalities and `evaluate_integrate()` which can be used to evaluate a method's performance on a prediction task. 

```python
# Parameters
# ----------------------------
# adata: Annotated data object
# x1key: string referring to the layer of first matrix in the AnnData object. Can be X, Ms, spliced, unspliced, velocity, or None
# x2key: tring referring to the layer of second matrix in the AnnData object. Can be X, Ms, spliced, unspliced, velocity, or None
# X1: matrix referring to the first data type if x1key unspecified
# X2: matrix referring to the second data type if x2key unspecified
# logX1: boolean referring to whether the first data type should be log transformed. If data type is Ms or velocity, this should be False.
# logX2: boolean referring to whether the second data type should be log transformed. If data type is Ms or velocity, this should be False.
# labels_key: string referring to the key in adata.obs of ground truth labels
# labels: array referring to the labels for every cell if labels_key is unspecified
# int_method: function that specifies the integration method to perform
# int_method_params: dictionary referring to the integration method hyperparameters
# eval_method: function that specifies the evaluation method to perform
# eval_method_params: dictionary referring to the evaluation method hyperparameters
# n_jobs: integer referring to the number of jobs to use in computation 

# Method attributes
# ----------------------------
# model.integrate()
#    performs integration of gene expression modalities

#    Returns:
#        W: sparse graph adjacency matrix of combined data
#        embed: embedding of combined data

# model.evaluate_integrate()
#    performs integration of gene expression modalities and then evaluates method according to the evaluation criteria or task of interest

#    Returns:
#        score_df: dataframe of classification or trajectory inferences scores

# Default
# ----------------------------

import evi
import scanpy as sc

adata = sc.read_h5ad('filename.h5ad')

model = evi.tl.EVI(adata = None, x1key = None, x2key = None, X1 = None, X2 = None, int_method = None,
            int_method_params = None, eval_method = None, eval_method_params = None, logX1 = None,
            logX2 = None, labels_key = None, labels = None, n_jobs = 1)
            
W, embed = model.integrate()
df = model.evaluate_integrate()
```

### Perform integration
Here we show you an example of how you can use the evi class to integrate gene expression modalities according to a multi-modal data integration strategy of interest. We provide 9 functions to perform integration. Alternatively, if you'd like to integrate your data according to another method of interest, feel free to add the function to the `merge.py` script.

* `evi.tl.expression` - unintegrated spliced gene expression data
* `evi.tl.concat_merge` - cell-wise (horizontal) concatenation of data modalities
* `evi.tl.sum_merge` - cell-wise sum of data modalities
* `evi.tl.cellrank` - implements [cellrank](https://www.doi.org/10.1038/s41592-021-01346-6) for merging moments of spliced and RNA velocity modalities
* `evi.tl.precise` - implements [PRECISE](https://www.doi.org/10.1093/bioinformatics/btz372) for merging gene expression modalities, where data modality one is projected onto the sorted principal vectors 
* `evi.tl.precise_consensus` - implements [PRECISE](https://www.doi.org/10.1093/bioinformatics/btz372) for merging gene expression modalities, where data modality one is projected onto the consensus features
* `evi.tl.snf` - implements [SNF](https://www.doi.org/10.1038/nmeth.2810) for merging gene expression modalities
* `evi.tl.grassmann` - implements [Grassmann joint embedding](https://www.doi.org/10.1093/bioinformatics/bty866) for merging gene expression modalities
* `evi.tl.integrated_diffusion` - implements [integrated diffusion](https://arxiv.org/pdf/2102.06757.pdf) for merging gene expression modalities

To perform integration, simply specify the data, whether an individual modality should be log transformed, the integration method, and the integration method hyperparameters. For more details on integration method-specific hyperparameters, please see the input parameters in the `merge.py` script.

```python
# Example integrating spliced and unspliced modalities using PRECISE

model = evi.tl.EVI(adata = adata, x1key = 'spliced', x2key = 'unspliced', logX1 = True, logX2 = True,
                   int_method = evi.tl.precise, int_method_params = {'n_pvs': 30}, n_jobs = -1)

W, embed = model.integrate()
```

### Evaluate trajectory inference
In this study, we evaluated the performance of multi-modal data integration strategies on combining gene expression modalities to infer a biologically meaningful trajectory. To do so, we compare predicted trajectories to a ground truth reference trajectory curated from the literature. If you'd like to perform trajectory inference evaluation, please follow the steps below.

#### Step 1: Construct ground truth trajectory
* If you have _a priori_ knowledge of known cellular transitions, you can construct a ground truth reference trajectory object for evaluation in [Dynverse](https://dynverse.org/) as follows.
* First define a milestone network of known cell type transitions.

```python

milestone_network = pd.DataFrame({"from": ["LTHSC_broad", "MPP_broad", "MPP_broad","LMPP_broad","CMP_broad","CMP_broad"],
                                  "to": ["MPP_broad", "CMP_broad", "LMPP_broad", "GMP_broad","MEP_broad","GMP_broad"],
                                  "length": np.ones(shape = (1, 6)).flatten(),
                                  "directed": np.repeat(True, 6)})

```
* Next, construct the ground truth reference trajectory h5ad object as, 

```python

# Parameters
# ----------------------------
# adata: annotated data object
# cluster_key: string referring to the ground truth labels key in adata object
# milestone_network: trajectory network consisting of groups and edges between them
# counts: raw count matrix of ground truth data. If None uses the adata.layers['raw_spliced'] layer
# expression: normalized and log transformed matrix of ground truth data. If None uses adata.X
# group_ids: series object of cluster names for every cell. If None, uses cluster_key
# cell_ids: series object of cell names. If None, uses adata.obs.index
# feature_ids: series object of feature names. If None, uses adata.var_names
# filename: string referring to the filename for saving

evi.tl.construct_ground_trajectory(adata, cluster_key = 'cell_types_broad_cleaned', milestone_network = milestone_network,
                                  counts = adata.layers['raw_spliced'], expression = adata.X, filename = 'gt_nestorowa')

```
#### Step 2: Add the ground truth trajectory
*  Once the trajectory object is created and saved, you can use it at any point. Simply load the object into the R environment prior to evaluation as,

```python
ground_trajectory = evi.tl.add_ground_trajectory('gt_nestorowa.h5ad')
```

#### Step 3: Perform trajectory inference evaluation
* In order to perform trajectory inference evaluation, you can use the `evaluate_integrate()` method in the evi class. 
* This will first perform integration according to the integration method and integration method hyperparameters specified.
* Next, it will perform evaluation according to the evaluation method and evaluation method parameters specified.
* Since we are interested in performing trajectory inference using PAGA + DPT and evaluating the inferred biological trajectory from integrated data, we will specify the evaluation method as `evi.tl.ti` and the evaluation method parameters to include the root cell or cluster, the number of diffusion map components, and the ground truth reference trajectory we created above.

```python

# Example for trajectory inference performance following moments of spliced and RNA velocity integration using SNF:

eval_method_params = {'root_cluster': 'LTHSC_broad', 'n_dcs': 20, 'connectivity_cutoff':0.05, 'root_cell': 646, 'ground_trajectory': ground_trajectory}

model = evi.tl.EVI(adata = adata, x1key = 'Ms', x2key = 'velocity',
                    logX1 = False, logX2 = False, labels_key = 'cell_types_broad_cleaned',
                    int_method = evi.tl.snf, int_method_params = {'k':10, 'mu':0.7, 'K': 50},
                    eval_method = evi.tl.ti, eval_method_params = eval_method_params, n_jobs = -1)

df = model.evaluate_integrate() #scores according to metrics in dynverse, where hmean refers to the trajectory inference correlation score of cell distance and feature importance score correlations
```

### Evaluate classification
If you'd like to evaluate an integration method's performance on perturbation or disease state classification, we provide two strategies for doing so: label propagation and support vector machine classification. Alternatively, if you'd like to provide your own classification function, feel free to add it to the `infer.py` script.

* To perform data integration followed by classification using label propagation, please specify (1) the evaluation method as `evi.tl.lp` and (2) the evaluation method parameters, including the ratio of training nodes to consider and the metrics to measure performance. 

```python

# Example for label propagation classification following moments of spliced and RNA velocity integration using concatenation:

model = evi.tl.EVI(adata = adata, x1key = 'Ms', x2key = 'velocity', logX1 = False, logX2 = False,
                   labels_key = 'condition_broad', int_method = evi.tl.concat_merge,
                   int_method_params = {'k': 10}, eval_method = evi.tl.lp,
                   eval_method_params = {'train_size': 0.5, 'random_state': 0, 'metrics': ['F1', 'balanced_accuracy', 'auc', 'precision', 'accuracy']}, n_jobs = -1)
                   
df = model.evaluate_integrate() #scores according to metrics in eval_method_params

```

* To perform data integration followed by classification using a support vector machine classifier, please specify (1) the evaluation method as `evi.tl.svm` and (2) the evaluation method parameters, including the metrics including the metrics to measure performance. 

```python

# Example for SVM classification using one data modality - spliced gene expression:

model = evi.tl.EVI(adata = adata, x1key = 'spliced', logX1 = True, labels_key = 'condition_broad',
                   int_method = evi.tl.expression, int_method_params = {'k': 10}, eval_method = evi.tl.svm,
                   eval_method_params = {'random_state': 0, 'metrics': ['F1', 'balanced_accuracy', 'auc', 'precision', 'accuracy']}, n_jobs = -1)

df = model.evaluate_integrate() #scores according to metrics in eval_method_params

```
