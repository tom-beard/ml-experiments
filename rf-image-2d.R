# init --------

library(tidyverse)
library(tidymodels)
library(doFuture)
library(ambient)

all_cores <- parallel::detectCores(logical = FALSE) - 1
registerDoFuture()
cl <- makeCluster(all_cores)
# note: if parallel processing isn't working, use plan(sequential) to revert to sequential

theme_set(theme_minimal())


# get data ----------------------------------------------------------------

grid_res <- 30

data_input <- long_grid(seq(1, 10, length.out = grid_res),
                        seq(1, 10, length.out = grid_res)) %>% 
  mutate(lum = (gen_spheres(x, y, frequency = 0.2) + 1) / 2 )


# eda ---------------------------------------------------------------------

data_input %>% plot(lum)
data_input %>%
  ggplot() +
  geom_histogram(aes(x = lum))
data_input %>%
  ggplot() +
  geom_tile(aes(x = x, y = y, fill = lum))

# data cleaning -----------------------------------------------------------

data_cleaned <- data_input %>% as_tibble()

# train/test split --------------------------------------------------------

set.seed(42)

test_prop <- 0.7
folds <- 10

data_split <- initial_split(data_cleaned, prop = test_prop)
data_train <- training(data_split)
data_test <- testing(data_split)
data_vfold <- vfold_cv(data_train, v = folds, repeats = 1)


# tuning hyperparameters --------------------------------------------------

rf_model <- 
  rand_forest(mtry = tune(), min_n = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("ranger")

rf_workflow <- 
  workflow() %>% 
  add_model(rf_model) %>% 
  add_formula(lum ~ .)

rf_param <- 
  rf_workflow %>% 
  parameters() %>% 
  update(mtry = mtry(range = c(1L, 2L)), min_n = min_n(range = c(2L, 20L)))

rf_grid <- grid_regular(rf_param, levels = c(2L, 5L))

plan(future::cluster, workers = cl)
rf_search <- tune_grid(rf_workflow, grid = rf_grid, resamples = data_vfold,
                       param_info = rf_param)
beepr::beep()

# select best hyperparameters ---------------------------------------------

rf_search %>% 
  autoplot(metric = "rmse")

show_best(rf_search, metric = "rmse", n = 9, maximize = FALSE)

# finalise model -------------------------------------------------------

rf_param_final <- select_by_one_std_err(rf_search, mtry, metric = "rmse", maximize = FALSE)

rf_workflow_finalized <- rf_workflow %>%
  finalize_workflow(rf_param_final)

# test predictions -------------------------------------------------------

rf_workflow_final_fit <- rf_workflow_finalized %>%
  fit(data = data_train)

data_test %>% 
  bind_cols(predict(rf_workflow_final_fit, new_data = data_test)) %>% 
  metrics(truth = lum, estimate = .pred)

# could use tune::last_fit() instead
# more concise, but doesn't return predictions joined to all cols of full test set
# also, to get the final fit and prepped recipes for vip, need to do
# rf_workflow_last_fit$.workflow[[1]]

rf_workflow_last_fit <- rf_workflow_finalized %>%
  last_fit(data_split, metrics = metric_set(rmse, rsq, mae))

rf_workflow_last_fit %>% collect_metrics()

