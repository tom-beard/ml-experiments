# from https://koalaverse.github.io/vip/articles/vip.html

library(tidymodels)

# simulate training data -------------------------------------------------------------

set.seed(101)  # for reproducibility
trn <- as.data.frame(mlbench::mlbench.friedman1(500))
# trn <- as_tibble(mlbench::mlbench.friedman1(500))


# trees & tree ensembles --------------------------------------------------------------------

# Load required packages
library(xgboost)  # for fitting GBMs
library(ranger)   # for fitting random forests
library(rpart)    # for fitting CART-like decision trees

# Fit a single regression tree
tree <- rpart(y ~ ., data = trn)

# Fit a random forest
set.seed(101)
rfo <- ranger(y ~ ., data = trn, importance = "impurity") # trn must be df, not tibble?

# Fit a GBM
set.seed(102)
bst <- xgboost(
  data = data.matrix(subset(trn, select = -y)),
  label = trn$y, 
  objective = "reg:linear",
  nrounds = 100, 
  max_depth = 5, 
  eta = 0.3,
  verbose = 0  # suppress printing
)

# to do: use parsnip idiom for the above



