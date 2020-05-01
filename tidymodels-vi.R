# init --------

library(tidymodels)
library(doFuture)
# library(xgboost)
library(vip)

theme_set(theme_light())

data("diamonds")

data_input <- diamonds

# train/test split --------------------------------------------------------

set.seed(1243)

# demo settings:
test_prop <- 0.1
folds <- 3
# more realistic settings: takes ~15 minutes on laptop?
# test_prop <- 0.7
# folds <- 10

# generic
data_split <- initial_split(data_input, prop = test_prop, strata = price)
data_train <- training(data_split)
data_test <- testing(data_split)

# depends on data structure and variable names
data_vfold <- vfold_cv(data_train, v = folds, repeats = 1, strata = price)

# feature engineering (including tuneable step) --------------------------

model_recipe <- 
  recipe(price ~ ., data = data_train) %>% 
  # need to explicitly specify recipes:: so that the functions can be found by parallel workers
  step_log(recipes::all_outcomes()) %>%
  step_normalize(recipes::all_predictors(), -recipes::all_nominal()) %>% 
  step_dummy(recipes::all_nominal()) %>% 
  step_poly(carat, degree = tune())

# tuning hyperparameters --------------------------------------------------

rf_model <- 
  rand_forest(mtry = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("ranger")

# workflows ---------------------------------------------------------------

rf_workflow <- 
  workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(model_recipe)

rf_param <- 
  rf_workflow %>% 
  parameters() %>% 
  update(mtry = mtry(range = c(3L, 5L)),
         degree = degree_int(range = c(2L, 4L)))

rf_grid <- grid_regular(rf_param, levels = 3)

all_cores <- parallel::detectCores(logical = FALSE) - 1
registerDoFuture()
cl <- makeCluster(all_cores)
plan(future::cluster, workers = cl)
# note: if parallel processing isn't working, use plan(sequential) to revert to sequential

rf_search <- tune_grid(rf_workflow, grid = rf_grid, resamples = data_vfold,
                       param_info = rf_param)
beepr::beep()

rf_search %>% 
  autoplot(metric = "rmse")
show_best(rf_search, metric = "rmse", n = 9, maximize = FALSE)
select_best(rf_search, metric = "rmse", maximize = FALSE)
select_by_one_std_err(rf_search, mtry, degree, metric = "rmse", maximize = FALSE)


# final predictions -------------------------------------------------------

rf_param_final <- select_by_one_std_err(rf_search, mtry, degree, metric = "rmse", maximize = FALSE)
rf_workflow_final <- rf_workflow %>% finalize_workflow(rf_param_final)
rf_workflow_final_fit <- rf_workflow_final %>% fit(data = data_train)

