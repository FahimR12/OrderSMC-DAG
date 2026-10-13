#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6L) {
  stop("usage: generate_clgaussian.R n_train n_test train_seed test_seed train.csv test.csv")
}

suppressPackageStartupMessages(library(bnlearn))
n_train <- as.integer(args[[1L]])
n_test <- as.integer(args[[2L]])
train_seed <- as.integer(args[[3L]])
test_seed <- as.integer(args[[4L]])
train_path <- args[[5L]]
test_path <- args[[6L]]

data(clgaussian.test)
model <- "[A][B][C][H][D|A:H][F|B:C][E|B:D][G|A:D:E:F]"
dag <- model2network(model)
fitted <- bn.fit(dag, clgaussian.test, method = "mle-cg")

encode <- function(frame) {
  for (node in names(frame)) {
    if (is.factor(frame[[node]])) {
      frame[[node]] <- as.integer(frame[[node]]) - 1L
    }
  }
  frame
}

set.seed(train_seed)
train <- encode(rbn(fitted, n = n_train))
set.seed(test_seed)
test <- encode(rbn(fitted, n = n_test))
write.csv(train, train_path, row.names = FALSE, quote = FALSE)
write.csv(test, test_path, row.names = FALSE, quote = FALSE)

