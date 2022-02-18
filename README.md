# EVI

Expression and Velocity Integration

## Introduction
<p align="center">
  <img src="/doc/pipeline.png"/>
</p>

## Installation

```
git clone https://github.com/jranek/EVI.git
```

## Dependencies

Creates an environment with most of the required packages.

```
conda env create -f venv_EVI.yml
source activate venv_EVI
```

A few python packages are unavailable through conda or pip. Upon creating the environment, we recommend that you install these manually. 

```
#precise v1.2
pip install --user git+https://github.com/NKI-CCB/PRECISE
```

For R packages, open R/ Jupyter magic within the environment and run the following commands:

```
install.packages("devtools")
install.packages("BiocManager")

#dynverse/dyncli v0.0.3.9000
devtools::install_github("dynverse/dyncli")

#dynverse/dyno v0.1.2
devtools::install_github("dynverse/dyno")

#dynverse/dyneval v0.9.9
devtools::install_github("dynverse/dyneval")

#dynverse/dyntoy v0.9.9
devtools::install_github("dynverse/dyntoy")

#SingleCellExperiment v1.14.1
BiocManager::install("SingleCellExperiment")

#batchelor v1.8.0
BiocManager::install("batchelor")

#scran v1.20.1
BiocManager::install("scran")
```

## Data access
Access the preprocessed loom or adata objects from [Zenodo](https://zenodo.org/record/6110279#.Yg1jPN_MK3C).

## Tutorial

### Perform preprocessing

### Perform integration

```
model = evi.tl.EVI(adata = adata, x1key = x1key, x2key = x2key, X1 = X1, X2 = X2, int_method = int_method,
            int_method_params = int_method_params, eval_method = eval_method, eval_method_params = eval_method_params, logX1 = logX1,
            logX2 = logX2, labels_key = labels_key, labels = labels, n_jobs = n_jobs)

W, embed = model.integrate()
```

### Evaluate trajectory inference

### Evaluate classification 
