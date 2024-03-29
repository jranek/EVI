seurat_v4 <- function(X1, X2, X1_pca, X2_pca, k){
    #merges data modalities using Seurat v4 FindMultiModalNeighbors function: https://www.sciencedirect.com/science/article/pii/S0092867421005833
    library("Seurat")
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
                                    k.nn = k)

    W <- as.data.frame(combined[["wknn"]])

    return(W)   
}