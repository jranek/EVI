library("Seurat")
library("future")

plan("multiprocess")

seurat_v4_rt <- function(k){

    X1 = read.csv('X1.csv', header = TRUE, row.names = 1)
    X2 = read.csv('X2.csv', header = TRUE, row.names = 1)
    X1_pca = read.csv('X1_pca.csv', header = TRUE, row.names = 1)
    X2_pca = read.csv('X2_pca.csv', header = TRUE, row.names = 1)

    tic = Sys.time()
    combined <- CreateSeuratObject(counts = t(X1), assay='x1')
    s2 <- CreateAssayObject(data = t(X2))

    combined[['x2']] <- s2

    X1_pca = as.matrix(X1_pca)
    X2_pca = as.matrix(X2_pca)
    
    combined[["x1_pca"]] <- CreateDimReducObject(embeddings = X1_pca, key = "PC_", assay = 'x1')
    combined[["x2_pca"]] <- CreateDimReducObject(embeddings = X2_pca, key = "PC_", assay = 'x2')

    combined <- FindMultiModalNeighbors(combined,
                                    reduction.list = list("x1_pca", "x2_pca"),
                                    dims.list = list(1:dim(X1_pca)[2], 1:dim(X2_pca)[2]),
                                    knn.range = 150,
                                    k.nn = k,
                                    verbose = FALSE)

    toc = Sys.time()
    elapsed = toc-tic
    return(elapsed)   
}