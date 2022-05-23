library("SingleCellExperiment")
library("dyneval")
library("dyno")
library("dyntoy")
library("dyncli")
library("tidyverse")

run_slingshot <- function(embedding, labels, root_cluster, add_noise){
    #### performs trajectory inference with Slingshot, then constructs trajectory graph for evaluation in dynverse:
    #### code pulled from dynverse container: https://github.com/dynverse/ti_slingshot/blob/master/package/R/ti_slingshot.R
    library("slingshot")
    library("igraph")

    parameters <- list()
    parameters$shrink <- 1L
    parameters$reweight <- TRUE
    parameters$reassign <- TRUE
    parameters$thresh <- 0.001
    parameters$maxit <- 10L
    parameters$stretch <- 2L
    parameters$smoother <- "smooth.spline"
    parameters$shrink.method <-"cosine"

    if (add_noise == TRUE) {
        span <- range(embedding)[2] - range(embedding)[1]
        N <- dim(embedding)[1]
        K <- dim(embedding)[2]

        noise <- rnorm(N*K, mean = 0, sd = span/100)
        dim(noise) <- c(N, K)

        embedding = embedding+noise
    }

    sds <- slingshot::slingshot(embedding,
                                labels, 
                                start.clus = root_cluster,
                                shrink = parameters$shrink,
                                reweight = parameters$reweight,
                                reassign = parameters$reassign,
                                thresh = parameters$thresh,
                                maxit = parameters$maxit,
                                stretch = parameters$stretch,
                                smoother = parameters$smoother,
                                shrink.method = parameters$shrink.method,
                                dist.method = 'slingshot',
                                approx_points = 100)

    # # satisfy r cmd check
    from <- to <- NULL

    # get distances between lineages

    # collect milestone network
    lineages <- slingLineages(sds)
    edge_weights <- E(slingMST(sds))$weight

    cluster_network <- lineages %>%
        map_df(~ tibble(from = .[-length(.)], to = .[-1])) %>%
        unique() %>%
        mutate(
        length = edge_weights, #lineage_ctrl$dist[cbind(from, to)]
        directed = TRUE
        )

    # # collect dimred
    dimred <- cellData(sds)$reducedDim

    # # collect clusters
    cluster <- cellData(sds)$clusterLabels

    # # collect progressions
    adj <- as.matrix(igraph::as_adj(slingMST(sds)), sparse = FALSE)

    lin_assign <- apply(slingCurveWeights(sds), 1, which.max)

    progressions <- map_df(seq_along(lineages), function(l) {
        ind <- lin_assign == l
        lin <- lineages[[l]]
        pst.full <- slingPseudotime(sds, na = FALSE)[,l]
        pst <- pst.full[ind]
        means <- sapply(lin, function(clID){
            stats::weighted.mean(pst.full, cluster[,clID])
        })
        non_ends <- means[-c(1,length(means))]
        edgeID.l <- as.numeric(cut(pst, breaks = c(-Inf, non_ends, Inf)))
        from.l <- lineages[[l]][edgeID.l]
        to.l <- lineages[[l]][edgeID.l + 1]
        m.from <- means[from.l]
        m.to <- means[to.l]

        pct <- (pst - m.from) / (m.to - m.from)
        pct[pct < 0] <- 0
        pct[pct > 1] <- 1

        tibble(cell_id = names(which(ind)), from = from.l, to = to.l, percentage = pct)
        })

    output <- list()
    output$milestone_network <- cluster_network
    output$progressions <- progressions

    return(output)
}

perform_evaluation_PAGA <- function(grouping, branch_progressions, branches, branch_network, filename){
    #### performs evaluation with trajectory graph generated from PAGA, if fails then returns NaNs

    #read in ground truth reference
    ground_trajectory <- dynutils::read_h5(paste0(filename, '.h5ad'))

    #convert df to matrix and make sparse :)
    ground_trajectory$counts <- as.matrix(ground_trajectory$counts, sparse = TRUE)
    ground_trajectory$expression <- as.matrix(ground_trajectory$expression, sparse = TRUE)

    #add the trajectory to the reference trajectory object  
    ground_trajectory <- add_trajectory(ground_trajectory,
                                        milestone_ids = ground_trajectory$milestone_ids,
                                        milestone_network = ground_trajectory$milestone_network,
                                        milestone_percentages = ground_trajectory$milestone_percentages)

    #construct the predicted trajectory object (has same expression, counts, feature names, cell names because it's the same data)
    infer_trajectory <- dynwrap::wrap_expression(cell_info = ground_trajectory$cell_info,
                                                feature_info = ground_trajectory$feature_info,
                                                counts =  ground_trajectory$counts, 
                                                expression = ground_trajectory$expression) 

    tryCatch(
        expr = {
            #add the predicted trajectory to the predicted trajectory object    
            infer_trajectory <- dynwrap::add_branch_trajectory(infer_trajectory,
                                                                grouping = grouping,
                                                                branch_progressions = branch_progressions,
                                                                branches = branches,
                                                                branch_network = branch_network) 

            #create waypoints, these are a subset of cells meant to speed up computation
            ground_trajectory <- add_cell_waypoints(ground_trajectory)
            infer_trajectory <- add_cell_waypoints(infer_trajectory)

            #compute metrics for evaluation
            eval_metrics <- dyneval::calculate_metrics(ground_trajectory, infer_trajectory,
                                                        metrics = c('correlation','featureimp_wcor'),
                                                        expression_source = ground_trajectory$expression)
            
            eval_metrics_hmean <- dynutils::calculate_harmonic_mean(eval_metrics$correlation, eval_metrics$featureimp_wcor)
            eval_metrics_gmean <- dynutils::calculate_geometric_mean(eval_metrics$correlation, eval_metrics$featureimp_wcor)

            eval_metrics <- as.data.frame(eval_metrics)
            eval_metrics$hmean <- eval_metrics_hmean
            eval_metrics$gmean <- eval_metrics_gmean
            eval_metrics <- t(eval_metrics)

            eval_output <- list()
            eval_output[['scores']] <- eval_metrics
            eval_output[['metric_labels']] <- rownames(eval_metrics)

            return(eval_output)
            },
            
        error = function(e){
            print(e)
            eval_output <- list()
            eval_output[['scores']] <- NaN
            eval_output[['metric_labels']] <- NaN
            return(eval_output)
        }
    )
}

perform_evaluation_angle <- function(pseudotime, filename){
    #### performs evaluation with trajectory graph generated from angle, if fails then returns NaNs

    #read in ground truth reference
    ground_trajectory <- dynutils::read_h5(paste0(filename, '.h5ad'))

    #convert df to matrix and make sparse :)
    ground_trajectory$counts <- as.matrix(ground_trajectory$counts, sparse = TRUE)
    ground_trajectory$expression <- as.matrix(ground_trajectory$expression, sparse = TRUE)

    #add the trajectory to the reference trajectory object  
    ground_trajectory <- add_trajectory(ground_trajectory,
                                        milestone_ids = ground_trajectory$milestone_ids,
                                        milestone_network = ground_trajectory$milestone_network,
                                        milestone_percentages = ground_trajectory$milestone_percentages)

    #construct the predicted trajectory object (has same expression, counts, feature names, cell names because it's the same data
    infer_trajectory <- dynwrap::wrap_expression(cell_info = ground_trajectory$cell_info,
                                                feature_info = ground_trajectory$feature_info,
                                                counts =  ground_trajectory$counts, 
                                                expression = ground_trajectory$expression) 

    tryCatch(
        expr = {
            
            pseudotime <- tibble(pseudotime)

            #add the predicted trajectory to the predicted trajectory object    
            infer_trajectory <- add_cyclic_trajectory(infer_trajectory, 
                                                        pseudotime = pseudotime,
                                                        do_scale_minmax = FALSE)

            #create waypoints, these are a subset of cells meant to speed up computation
            ground_trajectory <- add_cell_waypoints(ground_trajectory)
            infer_trajectory <- add_cell_waypoints(infer_trajectory)

            #compute metrics for evaluation
            eval_metrics <- dyneval::calculate_metrics(ground_trajectory, infer_trajectory,
                                                        metrics = c('correlation','featureimp_wcor'),
                                                        expression_source = ground_trajectory$expression)
            
            eval_metrics_hmean <- dynutils::calculate_harmonic_mean(eval_metrics$correlation, eval_metrics$featureimp_wcor)
            eval_metrics_gmean <- dynutils::calculate_geometric_mean(eval_metrics$correlation, eval_metrics$featureimp_wcor)

            eval_metrics <- as.data.frame(eval_metrics)
            eval_metrics$hmean <- eval_metrics_hmean
            eval_metrics$gmean <- eval_metrics_gmean
            eval_metrics <- t(eval_metrics)

            eval_output <- list()
            eval_output[['scores']] <- eval_metrics
            eval_output[['metric_labels']] <- rownames(eval_metrics)

            return(eval_output)
            },
            
        error = function(e){
            print(e)
            eval_output <- list()
            eval_output[['scores']] <- NaN
            eval_output[['metric_labels']] <- NaN
            return(eval_output)
        }
    )
}

perform_evaluation_slingshot <- function(embedding, labels, root_cluster, filename, add_noise){
    #### performs evaluation with trajectory graph generated from slingshot, if fails then returns NaNs

    #read in ground truth reference
    ground_trajectory <- dynutils::read_h5(paste0(filename, '.h5ad'))

    #convert df to matrix and make sparse :)
    ground_trajectory$counts <- as.matrix(ground_trajectory$counts, sparse = TRUE)
    ground_trajectory$expression <- as.matrix(ground_trajectory$expression, sparse = TRUE)

    #add the trajectory to the reference trajectory object  
    ground_trajectory <- add_trajectory(ground_trajectory,
                                        milestone_ids = ground_trajectory$milestone_ids,
                                        milestone_network = ground_trajectory$milestone_network,
                                        milestone_percentages = ground_trajectory$milestone_percentages)

    #construct the predicted trajectory object (has same expression, counts, feature names, cell names because it's the same data
    infer_trajectory <- dynwrap::wrap_expression(cell_info = ground_trajectory$cell_info,
                                                feature_info = ground_trajectory$feature_info,
                                                counts =  ground_trajectory$counts, 
                                                expression = ground_trajectory$expression) 

    tryCatch(
        expr = {
            
            #estimate pseudotime with slingshot
            output <- run_slingshot(embedding, labels, root_cluster,add_noise)
            milestone_network <- output$milestone_network
            progressions <- output$progressions

            #add the predicted trajectory to the predicted trajectory object    
            infer_trajectory <- add_trajectory(infer_trajectory, 
                                                milestone_network = milestone_network,
                                                progressions = progressions)

            #create waypoints, these are a subset of cells meant to speed up computation
            ground_trajectory <- add_cell_waypoints(ground_trajectory)
            infer_trajectory <- add_cell_waypoints(infer_trajectory)

            #compute metrics for evaluation
            eval_metrics <- dyneval::calculate_metrics(ground_trajectory, infer_trajectory,
                                                        metrics = c('correlation','featureimp_wcor'),
                                                        expression_source = ground_trajectory$expression)
            
            eval_metrics_hmean <- dynutils::calculate_harmonic_mean(eval_metrics$correlation, eval_metrics$featureimp_wcor)
            eval_metrics_gmean <- dynutils::calculate_geometric_mean(eval_metrics$correlation, eval_metrics$featureimp_wcor)

            eval_metrics <- as.data.frame(eval_metrics)
            eval_metrics$hmean <- eval_metrics_hmean
            eval_metrics$gmean <- eval_metrics_gmean
            eval_metrics <- t(eval_metrics)

            eval_output <- list()
            eval_output[['scores']] <- eval_metrics
            eval_output[['metric_labels']] <- rownames(eval_metrics)

            return(eval_output)
            },
            
        error = function(e){
            print(e)
            eval_output <- list()
            eval_output[['scores']] <- NaN
            eval_output[['metric_labels']] <- NaN
            return(eval_output)
        }
    )
}