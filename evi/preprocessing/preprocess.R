run_multiBatchNorm <- function(X1, X2, batches, clusters) {
    # performs per-batch scaling normalization with batchelor: https://www.nature.com/articles/nbt.4091
    # implemented in R: https://rdrr.io/github/LTLA/batchelor/man/multiBatchNorm.html

    # Parameters
    # X1: data.frame
    #   dataframe referring to spliced counts. Dimensions are (genes x cells)
    # X2: data.frame
    #   dataframe referring to unspliced counts. Dimensions are (genes x cells)
    # batches: data.frame
    #   dataframe referring to batch condition of every cell. Dimensions are (1 x cells)
    # clusters: data.frame
    #   dataframe referring to cluster annotation for every cell. Dimensions are (1 x cells)
    # ----------

    # Returns
    # eval_output: list
    #   list of size factors for spliced and unspliced counts according to batch
    #       eval_output[['size_factors_x']]
    #       eval_output[['size_factors_u']]

    # ----------
    library("SingleCellExperiment")
    library("batchelor")
    library("scran")

    clusters <- t(as.data.frame(unlist(clusters)))
    batches <- t(as.data.frame(unlist(batches)))

    X1 <- as.matrix(X1)
    x2 <- as.matrix(X2)

    size_factors_X1 <- calculateSumFactors(X1, clusters=clusters, min.mean=0.1) #calculate sum factors for matrix, compute for sce
    size_factors_X2 <- calculateSumFactors(X2, clusters=clusters, min.mean=0.1)

    sce_X1 <- SingleCellExperiment(list(counts=X1))
    sizeFactors(sce_X1) <- size_factors_X1
    sce_X2 <- SingleCellExperiment(list(counts=X2))  
    sizeFactors(sce_X2) <- size_factors_X2  

    out_X1 <- multiBatchNorm(sce_X1, batch = batches)
    out_X2 <- multiBatchNorm(sce_X2, batch = batches)

    size_factors_X1 <- sizeFactors(out_X1)
    size_factors_X2 <- sizeFactors(out_X2)

    eval_output <- list()
    eval_output[['size_factors_x']] <- size_factors_X1
    eval_output[['size_factors_u']] <- size_factors_X2

    return(eval_output)
}

run_scran <- function(X1, X2, clusters){
    # performs normalization with scran: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-0947-7
    # implemented in R: https://rdrr.io/bioc/scran/man/computeSumFactors.html

    # Parameters
    # X1: data.frame
    #   dataframe referring to spliced counts. Dimensions are (genes x cells)
    # X2: data.frame
    #   dataframe referring to unspliced counts. Dimensions are (genes x cells)
    # clusters: data.frame
    #   dataframe referring to cluster annotation for every cell. Dimensions are (1 x cells)
    # ----------

    # Returns
    # eval_output: list
    #   list of size factors for spliced and unspliced counts according to batch
    #       eval_output[['size_factors_x']]
    #       eval_output[['size_factors_u']]

    # ----------
    library("SingleCellExperiment")
    library("scran")

    clusters <- t(as.data.frame(unlist(clusters)))
    size_factors_X1 <- calculateSumFactors(X1, clusters=clusters, min.mean=0.1) #calculate sum factors for matrix, compute for sce
    size_factors_X2 <- calculateSumFactors(X2, clusters=clusters, min.mean=0.1)

    eval_output <- list()
    eval_output[['size_factors_x']] <- size_factors_X1
    eval_output[['size_factors_u']] <- size_factors_X2
    return(eval_output)
}

run_mnnCorrect <- function(X, batches){
    # performs batch effect correction on horizontally concatenated spliced and unspliced data using mutual nearest neighbors: https://www.nature.com/articles/nbt.4091
    # implemented in R: https://rdrr.io/bioc/batchelor/man/mnnCorrect.html

    # Parameters
    # X1: data.frame
    #   dataframe referring to spliced counts. Dimensions are (genes x cells)
    # X2: data.frame
    #   dataframe referring to unspliced counts. Dimensions are (genes x cells)
    # batches: data.frame
    #   dataframe referring to batch condition of every cell. Dimensions are (1 x cells)
    # ----------

    # Returns
    # eval_output: list
    #   list of size factors for spliced and unspliced counts according to batch
    #       eval_output[['size_factors_x']]
    #       eval_output[['size_factors_u']]

    # ----------
    library("SingleCellExperiment")
    library("batchelor")

    X <- as.matrix(X) #(2*genes x cells)

    X_c <- mnnCorrect(X, batch = batches, k = 10)
    X_c <- assay(X_c, "corrected")

    eval_output <- list()
    eval_output[['corrected']] <- X_c

    return(eval_output)
}