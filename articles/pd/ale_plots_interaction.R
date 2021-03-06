# Generate a single ALE plot per variable in `interaction` dataset

# Load the necessary packages
library(dplyr)
library(magrittr)
library(readr)
library(randomForest)
library(ALEPlot)

# Load, modify data ----------------------------------------------------------------------
df <- read_csv('interaction.csv')
X <- df %>% select(-y) %>% as.data.frame
y <- df$y

# Utils ----------------------------------------------------------------------------------
rf_predict <- function(X.model, newdata) {
  return (as.numeric(predict(X.model, newdata)))
}

# Fit RandomForest model to rent data -------------------------------------------------
rf <- randomForest(X, y, ntree=10)

# Make plots -----------------------------------------------------------------------------
make_plots <- function(X, features=names(X), intervals=rep(100, length(features)),
                       base_filename='images/', width=5, height=5) {
  # For each feature in X, generate an ALE plot and save as pdf
  for (i in 1:length(features)) {
    col_idx <- which(names(X) %in% features[i])
    K <- intervals[i]
    filename <- paste0(base_filename, features[i], '_ale.pdf')
    pdf(file=filename, width=width, height=height)
    message(paste0('Saving ', filename))
    ale <- ALEPlot(X, rf, pred.fun=rf_predict, J=col_idx, K=K)
    filename <- paste0(base_filename, features[i], '_ale.csv')
    write.csv(ale, filename, row.names=FALSE)
    message(paste0('Saved ', filename))
    dev.off()
  }
}

# Create ALE plot PDFs.  Here we can assign K to each feature.
make_plots(X, c('x1', 'x2', 'x3'), intervals=rep(300, 3))
