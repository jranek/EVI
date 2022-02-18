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

1. Create a conda environment with most of the required packages

```
conda env create -f venv_EVI.yml
```

2. Activate the environment 

```
source activate venv_EVI
```

3. Install the remaining dependencies

A few python and R dependencies are unavailable through conda or pip. To install them, run the following make command.

```
make
```

It's important to note that dynverse is a large collection of packages that requires many installs. If you run into install issues, follow their guidelines here: [dynverse](https://dynverse.org/users/1-installation/)

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
