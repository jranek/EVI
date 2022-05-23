chooseCRANmirror(ind = 77)

install.packages("devtools")
install.packages("BiocManager")

#SingleCellExperiment v1.14.1
BiocManager::install("SingleCellExperiment")

#batchelor v1.8.0
BiocManager::install("batchelor")

#slingshot v2.0.0
BiocManager::install("slingshot")

#scran v1.20.1
BiocManager::install("scran")

#dynverse/dyncli v0.0.3.9000
devtools::install_github("dynverse/dyncli")

#dynverse/dyno v0.1.2
devtools::install_github("dynverse/dyno")

#dynverse/dyneval v0.9.9
devtools::install_github("dynverse/dyneval")

#dynverse/dyntoy v0.9.9
devtools::install_github("dynverse/dyntoy")