library("SingleCellExperiment")
library("dyneval")
library("dyno")
library("dyntoy")
library("dyncli")

add_ground_trajectory <- function(directory, filename){

    #creates ground truth trajectory from saved object

    trajectory <- dynutils::read_h5(paste0(directory,'/', filename))

    trajectory$counts <- as.matrix(trajectory$counts, sparse = TRUE)
    trajectory$expression <- as.matrix(trajectory$expression, sparse = TRUE)
    
    trajectory <- add_trajectory(trajectory,
                                milestone_ids = trajectory$milestone_ids,
                                milestone_network = trajectory$milestone_network,
                                milestone_percentages = trajectory$milestone_percentages)
    
    return(trajectory)
}

perform_evaluation_PAGA <- function(grouping, branch_progressions, branches, branch_network, ground_trajectory){

    #creates inferred trajectory object
    #then performs evaluation in comparison to ground truth reference and computes metrics
    #if fails, returns nans 

    infer_trajectory <- dynwrap::wrap_expression(cell_info = ground_trajectory$cell_info,
                                                feature_info = ground_trajectory$feature_info,
                                                counts = ground_trajectory$counts,
                                                expression = ground_trajectory$expression)

    tryCatch(
    expr = {
    infer_trajectory <- dynwrap::add_branch_trajectory(infer_trajectory,
                                                        grouping = grouping,
                                                        branch_progressions = branch_progressions,
                                                        branches = branches,
                                                        branch_network = branch_network) 

    ground_trajectory <- add_cell_waypoints(ground_trajectory)
    infer_trajectory <- add_cell_waypoints(infer_trajectory)

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
        eval_output <- list()
        eval_output[['scores']] <- NaN
        eval_output[['metric_labels']] <- NaN
        return(eval_output)
    }
    )
}