# Generate a single ALE plot per variable in `weather` dataset

# Load the necessary packages
library(dplyr)
library(magrittr)
library(readr)
library(randomForest)
library(ALEPlot)

# Load, modify data ----------------------------------------------------------------------
df_weather <- read_csv('weather.csv')
df_weather$state <- as.factor(df_weather$state)  # Convert `state` to factor (categorical)

# Utils ----------------------------------------------------------------------------------
rf_predict <- function(X.model, newdata) {
  return (as.numeric(predict(X.model, newdata)))
}

# Fit RandomForest model to weather data -------------------------------------------------
X <- df_weather %>% select(-temperature) %>% as.data.frame
y <- df_weather$temperature
# set.seed(3)
rf_weather <- randomForest(X, y, ntree=150, nodesize=5, mtry=3)  # hyperparams found using gridsearch

# Make plots -----------------------------------------------------------------------------
make_plots <- function(X, features=c('dayofyear', 'state'),
                       intervals=rep(100, length(features)), base_filename='images/',
                       width=5, height=5) {
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
    filename <- paste0(base_filename, features[i], '_', K, '_ale.pdf')
    #pdf(file=filename, width=width, height=height)
    #message(paste0('Saving ', filename))
    ale <- ALEPlot(X, rf_weather, pred.fun=rf_predict, J=col_idx, K=K)
    filename <- paste0(base_filename, features[i], '_', K, '_ale.csv')
    write.csv(ale, filename, row.names=FALSE)
    message(paste0('Saved ', filename))
    dev.off()
  }
}

# Create ALE plot PDFs.  Here we can assign K to each feature.
# e.g. c('dayofyear', 'state') with intervals c(100, 5)
# uses K = 100 for 'dayofyear', 5 for 'state'
# Note that K for categorical variables is currently ignored by ALEPlot.
make_plots(X, c('dayofyear', 'state'), intervals = c(100, 5))
