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

# eda ---------------------------------------------------------------------

data_input %>% 
  ggplot() +
  geom_histogram(aes(x = hwy))

data_input %>% 
  ggplot() +
  geom_jitter(aes(x = hwy, y = cty, colour = class))

data_input %>% select_if(is.numeric) %>% corrr::correlate() %>% corrr::rplot()


# data cleaning -----------------------------------------------------------

data_cleaned <- data_input %>% 
  select(-cty)

# train/test split --------------------------------------------------------

set.seed(42)

# demo settings:
test_prop <- 0.1
folds <- 3
# more realistic settings: 
# test_prop <- 0.7
# folds <- 10

# set stratification, if required
data_split <- initial_split(data_cleaned, prop = test_prop)
# data_split <- initial_split(data_cleaned, prop = test_prop, strata = hwy)
data_train <- training(data_split)
data_test <- testing(data_split)
data_vfold <- vfold_cv(data_train, v = folds, repeats = 1)
# data_vfold <- vfold_cv(data_train, v = folds, repeats = 1, strata = hwy)

# feature engineering (potentially including tuneable steps) --------------------------

model_recipe <- 
  recipe(hwy ~ ., data = data_train) %>% 
  update_role(model, new_role = "model id") %>% 
  # need to explicitly specify recipes:: so that the functions can be found by parallel workers
  step_log(recipes::all_outcomes()) %>%
  step_normalize(recipes::all_predictors(), -recipes::all_nominal()) %>% 
  step_dummy(recipes::all_nominal())
