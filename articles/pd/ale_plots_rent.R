# Generate a single ALE plot per variable in `weight` dataset

# Load the necessary packages
library(dplyr)
library(magrittr)
library(readr)
library(randomForest)
library(ALEPlot)

# Load, modify data ----------------------------------------------------------------------
df_rent <- read_csv('rent10k.csv')
X <- df_rent %>% select(-price) %>% as.data.frame
y <- df_rent$price

# Utils ----------------------------------------------------------------------------------
rf_predict <- function(X.model, newdata) {
  return (as.numeric(predict(X.model, newdata)))
}

# Fit RandomForest model to rent data -------------------------------------------------
rf_rent <- randomForest(X, y, ntree=200, nodesize=5, mtry=5) # hyperparams found using gridsearch

# Make plots -----------------------------------------------------------------------------
make_plots <- function(X, features=names(X), intervals=rep(100, length(features)),
                       base_filename='images/', width=5, height=5) {
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
    col_idx <- which(names(X) %in% features[i])
    K <- intervals[i]
    filename <- paste0(base_filename, features[i], '_ale.pdf')
    #pdf(file=filename, width=width, height=height)
    #message(paste0('Saving ', filename))
    ale <- ALEPlot(X, rf_rent, pred.fun=rf_predict, J=col_idx, K=K)
    filename <- paste0(base_filename, features[i], '_ale.csv')
    write.csv(ale, filename, row.names=FALSE)
    message(paste0('Saved ', filename))
    dev.off()
  }
}

# Create ALE plot PDFs.  Here we can assign K to each feature.
make_plots(X, c('bedrooms', 'bathrooms', 'longitude', 'latitude'), intervals=rep(300, 4))
