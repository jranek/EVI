# EVI
Expression and Velocity Integration

## Introduction
EVI is a python package designed for evaluating multi-modal data integration strategies on combining unspliced, spliced, and RNA velocity gene expression modalities for trajectory inference and disease prediction tasks. For more details on the methods considered and the overall benchmarking pipeline, please read the associated preprint:

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

```
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

This class provides two methods following instantiation, `integrate()` which can be used to integrate gene expression modalities and `evaluate_integrate` which can be used to evaluate a method's performance on a prediction task. 

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

# Attributes
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

model = evi.tl.EVI(adata = None, x1key = None, x2key = None, X1 = None, X2 = None, int_method = None,
            int_method_params = None, eval_method = None, eval_method_params = None, logX1 = None,
            logX2 = None, labels_key = None, labels = None, n_jobs = 1)
            
W, embed = model.integrate()
df = model.evaluate_integrate()
```

### Perform integration
Here we show you an example of how you can integrate gene expression modalities according to a multi-modal data integration strategy of interest. To do so, you must specify the data, whether the data should be log transformed, the integration method, and the integration method hyperparameters. For more details on integration method-specific hyperparameters, please see the input parameters in the `evi.tl.merge` script

We provide 9 functions to perform integration

* `evi.tl.expression`- unintegrated spliced gene expression data
* `evi.tl.concat_merge` - horizontal (cell-wise) concatenation of data modalities
* `evi.tl.sum_merge` - cell-wise sum of data modalities
* `evi.tl.cellrank` - implements cellrank merging of moments of spliced and RNA velocity modalities
* `evi.tl.precise` - implements PRECISE merging, where data modality 1 is projected onto the sorted principal vectors 
* `evi.tl.precise_consensus` - implements PRECISE merging, where data modality 1 is projected onto the consensus features of both modalities
* `evi.tl.snf` - merges data modalities using Similarity Network Fusion (SNF)
* `evi.tl.grassmann` - merges data modalities according to subspace analysis on a Grassmannian manifold
* `evi.tl.integrated_diffusion` - merges data modalities using integrated diffusion 

Alternatively, if you'd like to integrate your data according to another method of interest, feel free to add the function to the `evi.tl.merge` script

```python
# Example integrating spliced and unspliced modalities using PRECISE

model = evi.tl.EVI(adata = adata, x1key = 'spliced', x2key = 'unspliced', logX1 = True, logX2 = True,
                   int_method = evi.tl.precise, int_method_params = {'n_pvs': 30}, n_jobs = -1)

W, embed = model.integrate()
```

### Evaluate trajectory inference

### Evaluate classification 
