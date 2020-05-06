# init --------

library(tidymodels)
library(doFuture)
library(xgboost)
library(vip)
library(pdp)

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
# test_prop <- 0.1
# folds <- 5
# more realistic settings: 
test_prop <- 0.7
folds <- 10

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

rf_workflow <- 
  workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(model_recipe)

rf_param <- 
  rf_workflow %>% 
  parameters() %>% 
  update(mtry = mtry(range = c(3L, 11L)))

rf_grid <- grid_regular(rf_param, levels = 9)

plan(future::cluster, workers = cl)
rf_search <- tune_grid(rf_workflow, grid = rf_grid, resamples = data_vfold,
                       param_info = rf_param)
beepr::beep()

# select best hyperparameters ---------------------------------------------

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


# finalise model and test predictions -------------------------------------------------------

rf_param_final <- select_by_one_std_err(rf_search, mtry, metric = "rmse", maximize = FALSE)
# rf_param_final <- best_1se

# change model spec to include rf-specific variable importance metrics
rf_model_importance <- 
  rand_forest(mtry = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("ranger", importance = "impurity")

rf_workflow_final_fit <- rf_workflow %>%
  update_model(rf_model_importance) %>% 
  finalize_workflow(rf_param_final) %>%
  fit(data = data_train)

data_test %>% 
  bind_cols(
    predict(rf_workflow_final_fit, new_data = data_test)
    ) %>% 
  metrics(truth = hwy, estimate = .pred)

# variable importance -----------------------------------------------------

rf_final_fit <- pull_workflow_fit(rf_workflow_final_fit)
model_recipe_prepped <- pull_workflow_prepped_recipe(rf_workflow_final_fit)

p1 <- rf_final_fit %>% 
  vip(aesthetics = list(fill = "grey50")) + 
  ggtitle("rf, model-based")

# setting up model-agnostic vip seems to require:
# explicit call to vi_permute
# baked version of training data, since it's not in the fit object pulled from the workflow
# pred_wrapper to specify newdata

rf_vi_permute <- rf_final_fit %>% 
  vi_permute(
    train = bake(model_recipe_prepped, new_data = data_train),
    target = "hwy", 
    metric = "rsquared",
    nsim = 10,
    keep = TRUE,
    pred_wrapper = function(object, newdata) predict(object, newdata)
  )

p2 <- rf_vi_permute %>%
  vip(all_permutations = TRUE, aesthetics = list(fill = "grey50")) + 
  ggtitle("rf, permutation")

grid.arrange(p1, p2, ncol = 2)


# partial dependence plots ------------------------------------------------

features <- rf_vi_permute %>% top_n(10, wt = Importance) %>% pull(Variable)
features <- rf_workflow_final_fit$fit$fit$preproc$x_var

pdps <- lapply(features, FUN = function(feature) {
  pd <- partial(rf_final_fit$fit, pred.var = feature,
                train = bake(model_recipe_prepped, new_data = data_train))
  autoplot(pd) + 
    ylim(range(data_train$hwy)) +
    geom_point()
})
grid.arrange(grobs = pdps, ncol = 5)

# works okay, but labels are in baked units: how to we show these as raw?


# interactions ------------------------------------------------------------

# see https://bgreenwell.github.io/pdp/articles/pdp.html#multi-predictor-pdps

