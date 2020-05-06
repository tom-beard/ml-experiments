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
  skimr::skim()

data_input %>% 
  ggplot() +
  geom_histogram(aes(x = hwy))

data_input %>% 
  ggplot() +
  geom_jitter(aes(x = hwy, y = cty, colour = class))

data_input %>% select_if(is.numeric) %>% corrr::correlate() %>% corrr::rplot()


# data cleaning -----------------------------------------------------------

data_cleaned <- data_input %>% 
  select(-cty) %>% 
  mutate_if(is.character, factor) # need to make characters into factors before splitting/folding

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
  update_role(model, new_role = "quasi id") %>% 
  # need to explicitly specify recipes:: so that the functions can be found by parallel workers
  step_normalize(recipes::all_predictors(), -recipes::all_nominal()) %>% 
  step_dummy(recipes::all_nominal(), -recipes::has_role("quasi id")) # need to exclude ID-type columns

# check recipe results (optional)
model_recipe %>% summary()
model_recipe %>% prep() %>% juice()


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
  update(mtry = mtry(range = c(3L, 9L)))

rf_grid <- grid_regular(rf_param, levels = 7)

plan(future::cluster, workers = cl)
rf_search <- tune_grid(rf_workflow, grid = rf_grid, resamples = data_vfold,
                       param_info = rf_param)
beepr::beep()

rf_search %>% 
  autoplot(metric = "rmse")

show_best(rf_search, metric = "rmse", n = 9, maximize = FALSE)
best_metric <- select_best(rf_search, metric = "rmse", maximize = FALSE) %>% pull(mtry)
best_1se <- select_by_one_std_err(rf_search, mtry, metric = "rmse", maximize = FALSE) %>% pull(mtry)

# show metric with standard error and selected best values
rf_search %>%
  collect_metrics() %>% 
  filter(.metric == "rmse") %>% 
  ggplot(aes(x = mtry, y = mean)) +
  geom_linerange(aes(ymin = mean - std_err, ymax = mean + std_err)) +
  geom_point(aes(size = 2 + mtry %in% c(best_metric, best_1se),
                 colour = factor(mtry == best_1se))) +
  scale_colour_manual(values = c("TRUE" = "firebrick", "FALSE" = "grey50")) +
  theme(legend.position = "none")
