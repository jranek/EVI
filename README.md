# EVI

Expression and Velocity Integration

## Introduction

## Pipeline overview
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
    * Open R and generate a GitHub token with `usethis::create_github_token()`
    * Copy the GitHub token and set your personal access token within R using the `usethis::edit_r_environ()` command. This will open a tab for which you can edit the GITHUB_PAT variable with your token, as `GITHUB_PAT = 'your_github_token'` 
    * Close R
* To finish installation with the makefile, run

```
make
```

## Data access
You can download all of the preprocessed loom and h5ad files from the [Zenodo](https://zenodo.org/record/6110279#.Yg1jPN_MK3C) repository. This can be done directly from the website or through the terminal. 

To list all of the publicly available files for download, 

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

If you'd like to download an example dataset from the terminal, please specify both the zenodo url `https://zenodo.org/` and the dataset string identifier from above. 

```
curl https://zenodo.org/record/6110279/files/adata_lane.h5ad?download=1 --output adata_lane.h5ad
```

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
