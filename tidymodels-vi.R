# init --------

library(tidymodels)
library(doFuture)
library(xgboost)
library(vip)

all_cores <- parallel::detectCores(logical = FALSE) - 1
registerDoFuture()
cl <- parallel::makeCluster(all_cores)
# note: if parallel processing isn't working, use plan(sequential) to revert to sequential

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

# feature engineering (including tunable step) --------------------------

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

# note: `parameters.workflow()` was deprecated in tune 0.1.6.9003.
# Please use `hardhat::extract_parameter_set_dials()` instead. 

rf_grid <- grid_regular(rf_param, levels = 3)

plan(future::cluster, workers = cl)
rf_search <- tune_grid(rf_workflow, grid = rf_grid, resamples = data_vfold,
                       param_info = rf_param)
beepr::beep()

rf_search %>% 
  autoplot(metric = "rmse")
show_best(rf_search, metric = "rmse", n = 9)
select_best(rf_search, metric = "rmse")
select_by_one_std_err(rf_search, mtry, degree, metric = "rmse")


# final predictions -------------------------------------------------------

rf_param_final <- select_by_one_std_err(rf_search, mtry, degree, metric = "rmse")

# change model spec to include rf-specific variable importance metrics
rf_model_importance <- 
  rand_forest(mtry = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("ranger", importance = "impurity")

rf_workflow_final_fit <- rf_workflow %>%
  update_model(rf_model_importance) %>% 
  finalize_workflow(rf_param_final) %>%
  fit(data = data_train)

# Now, we want to use this to predict() on data never seen before, namely,
# data_test. Unfortunately, predict(rf_wflow_final_fit, new_data = data_test) does
# not work in the present case, because the outcome is modified in the recipe
# via step_log().
#
# Thus, we need a little workaround: The prepped recipe is extracted from the
# workflow, and this can then be used to bake() the testing data. This baked
# data set together with the extracted model can then be used for the final
# predictions.

model_recipe_prepped <- extract_recipe(rf_workflow_final_fit)
rf_final_fit <- extract_fit_parsnip(rf_workflow_final_fit)

data_test_transformed <- predict(rf_final_fit, new_data = bake(model_recipe_prepped, data_test)) %>% 
  bind_cols(data_test) %>% 
  mutate(log_price = log(price))

metrics(data_test_transformed, truth = log_price, estimate = .pred)

# there's probably a less case-specific workaround for this, but let's move on


# variable importance -----------------------------------------------------

p1 <- vip(rf_final_fit) + 
  ggtitle("rf, model-based")

# setting up model-agnostic vip seems to require:
# explicit call to vi_permute
# baked version of training data, since it's not in the fit object pulled from the workflow
# pred_wrapper to specify newdata

# this doesn't work: vi uses vi.model_fit, which wraps the call to vi_permute and tries to extract object$fit first
# model_agnostic_vi2 <- vi(rf_final_fit, method = "permute", target = "price",
#     train = bake(model_recipe_prepped, new_data = data_train),
#     metric = "rsquared", pred_wrapper = function(object, newdata) predict(object, newdata))

p2 <- vi_permute(rf_final_fit, target = "price", 
                 train = bake(model_recipe_prepped, new_data = data_train), metric = "rsquared",
                 pred_wrapper = function(object, newdata) predict(object, newdata)) %>%
  vip() + 
  ggtitle("rf, permutation")

grid.arrange(p1, p2, ncol = 2)
