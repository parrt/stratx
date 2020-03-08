# Generate a single ALE plot per variable in `weight` dataset

# Load the necessary packages
library(dplyr)  # I had to do: install.packages("dplyr")
library(magrittr)
library(readr)  # I had to do: install.packages("readr")
library(randomForest)  # I had to do: install.packages("randomForest")
library(ALEPlot)  # I had to do: install.packages("ALEPlot")

# Load, modify data ----------------------------------------------------------------------
df_weight <- read_csv('weight.csv') %>% select(-X1)  # Load df, drop X1 column
df_weight$sex <- as.factor(df_weight$sex)
df_weight$pregnant <- as.factor(df_weight$pregnant)

# Utils ----------------------------------------------------------------------------------
rf_predict <- function(X.model, newdata) {
  return (as.numeric(predict(X.model, newdata)))
}

# Fit RandomForest model to weight data -------------------------------------------------
X <- df_weight %>% select(-weight) %>% as.data.frame
y <- df_weight$weight
# set.seed(3)
rf_weight <- randomForest(X, y, ntree=500, nodesize=1, mtry=3) # hyperparams found using gridsearch

# Make plots -----------------------------------------------------------------------------
make_plots <- function(X, features=names(X), intervals=rep(100, length(features)),
                       base_filename='weight_', width=5, height=5) {
  # Generate ALE plots for the specified variables and save each plot to PDF
  # Saves PDF to current working directory.
  #
  # X: data frame of features
  # features: vector of feature names
  # intervals: vector corresponding to the K used for each feature
  # base_filename: filename prefix for pdf
  # width, height: pdf dimensions, in inches
  
  # Check that length of K is same as length of features
  stopifnot(length(features) == length(intervals))
  
  # For each feature in X, generate an ALE plot and save as pdf
  for (i in 1:length(features)) {
    col_idx <- which(names(X) == features[i])
    K <- intervals[i]
    filename <- paste0(base_filename, features[i], '_', K, '.pdf')
    pdf(file=filename, width=width, height=height)
    message(paste0('Saving ', filename))
    ALEPlot(X, rf_weight, pred.fun=rf_predict, J=col_idx, K=K)
    dev.off()
  }
}

# Create ALE plot PDFs.  Here we can assign K to each feature.
# e.g. c('height', 'education') with intervals c(100, 200)
# uses K = 100 for 'height', 200 for 'education'
# Note that K for categorical variables is currently ignored by ALEPlot.
make_plots(X, names(X), intervals=rep(100, ncol(X)))
