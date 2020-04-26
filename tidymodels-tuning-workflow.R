# init --------

# based on https://hansjoerg.me/2020/02/09/tidymodels-for-machine-learning/
# simplified to focus on tuning workflow

library(tidymodels)
library(doFuture)

theme_set(theme_light())

data("diamonds")

# train/test split --------------------------------------------------------

set.seed(1243)

# demo settings:
# test_prop <- 0.1
# folds <- 3

# more realistic settings: takes ~15 minutes on laptop?
test_prop <- 0.7
folds <- 10

dia_split <- initial_split(diamonds, prop = test_prop, strata = price)
dia_train <- training(dia_split)
dia_test <- testing(dia_split)

dia_vfold <- vfold_cv(dia_train, v = folds, repeats = 1, strata = price)


# feature engineering (including tuneable step) --------------------------

dia_recipe <- 
  recipe(price ~ ., data = dia_train) %>% 
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
  add_recipe(dia_recipe)

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

# note: if parallel processing isn't working, use the following line to revert to sequential
# plan(sequential)

rf_search <- tune_grid(rf_workflow, grid = rf_grid, resamples = dia_vfold,
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
rf_workflow_final_fit <- rf_workflow_final %>% fit(data = dia_train)

# Now, we want to use this to predict() on data never seen before, namely,
# dia_test. Unfortunately, predict(rf_wflow_final_fit, new_data = dia_test) does
# not work in the present case, because the outcome is modified in the recipe
# via step_log().
#
# Thus, we need a little workaround: The prepped recipe is extracted from the
# workflow, and this can then be used to bake() the testing data. This baked
# data set together with the extracted model can then be used for the final
# predictions.

dia_recipe_prepped <- pull_workflow_prepped_recipe(rf_workflow_final_fit)
rf_final_fit <- pull_workflow_fit(rf_workflow_final_fit)

dia_test$.pred <- predict(rf_final_fit, new_data = bake(dia_recipe_prepped, dia_test))$.pred
dia_test$log_price <- log(dia_test$price)

metrics(dia_test, truth = log_price, estimate = .pred)
