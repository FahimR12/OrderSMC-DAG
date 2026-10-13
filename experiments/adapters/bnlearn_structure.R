#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 8L) {
  stop("usage: bnlearn_structure.R data.csv types.tsv scores.tsv mode algorithm maxp seed output.tsv")
}

data_path <- args[[1L]]
types_path <- args[[2L]]
scores_path <- args[[3L]]
mode <- args[[4L]]
algorithm <- args[[5L]]
maxp <- as.integer(args[[6L]])
seed <- as.integer(args[[7L]])
output_path <- args[[8L]]

suppressPackageStartupMessages(library(bnlearn))
set.seed(seed)
data <- read.csv(data_path, check.names = FALSE)
types <- read.delim(types_path, stringsAsFactors = FALSE, check.names = FALSE)
if (!identical(sort(names(data)), sort(types$node))) {
  stop("type metadata do not match data columns")
}
for (index in seq_len(nrow(types))) {
  node <- types$node[[index]]
  if (types$type[[index]] != "con") {
    data[[node]] <- factor(data[[node]])
  } else {
    data[[node]] <- as.numeric(data[[node]])
  }
}

if (mode == "native-bic-cg") {
  score_name <- "bic-cg"
  score_args <- list()
} else if (mode == "shared-mixed-score") {
  score_rows <- read.delim(
    scores_path,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    colClasses = c("character", "character", "numeric")
  )
  score_environment <- new.env(hash = TRUE, parent = emptyenv())
  for (index in seq_len(nrow(score_rows))) {
    assign(
      paste(score_rows$node[[index]], score_rows$parents[[index]], sep = "\t"),
      score_rows$score[[index]],
      envir = score_environment
    )
  }
  local_score <- function(node, parents, data, args) {
    parent_key <- paste(sort(parents), collapse = "|")
    key <- paste(node, parent_key, sep = "\t")
    if (!exists(key, envir = args$scores, inherits = FALSE)) {
      return(-Inf)
    }
    get(key, envir = args$scores, inherits = FALSE)
  }
  score_name <- "custom-score"
  score_args <- list(fun = local_score, args = list(scores = score_environment))
} else {
  stop("unknown scoring mode")
}

common <- c(list(x = data, score = score_name, maxp = maxp), score_args)
if (algorithm == "hc") {
  fitted <- do.call(
    hc,
    c(common, list(restart = 0L, perturb = 0L, optimized = TRUE))
  )
} else if (algorithm == "tabu") {
  fitted <- do.call(
    tabu,
    c(common, list(tabu = 10L, max.tabu = 10L, max.iter = 10000L, optimized = TRUE))
  )
} else {
  stop("unknown search algorithm")
}

network_score <- do.call(score, c(list(x = fitted, data = data, type = score_name), score_args))
arc_table <- arcs(fitted)
output <- data.frame(
  record = c("metadata", rep("arc", nrow(arc_table))),
  parent = c("", if (nrow(arc_table)) arc_table[, "from"] else character()),
  child = c("", if (nrow(arc_table)) arc_table[, "to"] else character()),
  value = c(as.character(network_score), rep("", nrow(arc_table))),
  stringsAsFactors = FALSE
)
write.table(output, output_path, sep = "\t", row.names = FALSE, quote = FALSE)
