# init --------

library(tidymodels)
library(doFuture)
library(xgboost)
library(vip)

all_cores <- parallel::detectCores(logical = FALSE) - 1
registerDoFuture()
cl <- makeCluster(all_cores)
# note: if parallel processing isn't working, use plan(sequential) to revert to sequential

theme_set(theme_minimal())


# get data ----------------------------------------------------------------

data_input <- mpg %>%
  as_tibble()

# train/test split --------------------------------------------------------

set.seed(42)

# demo settings:
test_prop <- 0.1
folds <- 3
# more realistic settings: 
# test_prop <- 0.7
# folds <- 10

# need to set stratification, if required
data_split <- initial_split(data_input, prop = test_prop)
# data_split <- initial_split(data_input, prop = test_prop, strata = hwy)
data_train <- training(data_split)
data_test <- testing(data_split)
data_vfold <- vfold_cv(data_train, v = folds, repeats = 1)
# data_vfold <- vfold_cv(data_train, v = folds, repeats = 1, strata = hwy)

# feature engineering (potentially including tuneable steps) --------------------------

