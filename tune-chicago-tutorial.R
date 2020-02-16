
# from tutorial -----------------------------------------------------------

library(tidymodels)

chi_rec <- recipe(ridership ~ ., data = this_dataset) %>% 
  step_holiday(date) %>% 
  step_date(date) %>% 
  step_rm(date) %>% 
  step_dummy(all_nominal()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(one_of(!!stations)) %>% 
  step_pca(one_of(!!stations), num_comp = 10) %>% 
  step_normalize(all_predictors())

library(parsnip)

knn_mod <- nearest_neighbor(neighbors = 5) %>% 
  set_engine("kknn") %>% 
  set_mode("regression")

# fitted <- knn_mod %>% 
#   fit(ridership ~ Monroe + humidity, data = Chicago)

library(tune)

knn_mod <- nearest_neighbor(
  neighbors = tune(),
  weight_func = tune()
  ) %>% 
  set_engine("kknn") %>% 
  set_mode("regression")

chi_rec <- recipe(ridership ~ ., data = this_dataset) %>% 
  step_holiday(date) %>% 
  step_date(date) %>% 
  step_rm(date) %>% 
  step_dummy(all_nominal()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(one_of(!!stations)) %>% 
  step_pca(one_of(!!stations), num_comp = tune()) %>% 
  step_normalize(all_predictors())

time_resamples <- rolling_origin(this_dataset,
                                 initial = 364 * 15,
                                 assess = 7 * 4,
                                 skip = 7 * 4,
                                 cumulative = FALSE)

set.seed(2132)
grid_results <- tune_grid(chi_rec,
                          model = knn_mod,
                          resamples = time_resamples,
                          grid = 20)

collect_metrics(grid_results)
collect_metrics(grid_results, summarize = FALSE)
show_best(grid_results, metric = "rmse", maximize = FALSE)
good_values <- select_best(grid_results, metric = "rmse", maximize = FALSE)
new_rec <- finalize_recipe(chi_rec, good_values)
new_mod <- finalize_model(knn_mod, good_values)

grid_results %>%
  autoplot(metric = "rmse")


# as quasi-workflow -------------------------------------------------------

library(tidymodels)
library(parsnip)
library(tune)

# 0. obtain and clean data
this_dataset <- Chicago

# 0.5 test/train split!

# 1. define recipe
this_recipe <- recipe(ridership ~ ., data = this_dataset) %>%
  step_holiday(date) %>%
  step_date(date) %>%
  step_rm(date) %>%
  step_dummy(all_nominal()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(one_of(!!stations)) %>%
  step_pca(one_of(!!stations), num_comp = tune()) %>%
  step_normalize(all_predictors())

# 2. define model
this_model <- nearest_neighbor(neighbors = tune(),
                              weight_func = tune()) %>%
  set_engine("kknn") %>%
  set_mode("regression")

# 3. define resamples (different for independent obs)
set.seed(2132) # or earlier
time_resamples <- rolling_origin(this_dataset,
  initial = 364 * 15, assess = 7 * 4, skip = 7 * 4, cumulative = FALSE)

# 4. define tuning (usinf defaults?) then fit all versions for tuning
grid_results <- tune_grid(this_recipe,
                          model = this_model,
                          resamples = time_resamples,
                          grid = 20)

grid_metrics <- collect_metrics(grid_results) # can then do custom vis
grid_results %>% autoplot(metric = "rmse") # marginal plots: not recommended

show_best(grid_results, metric = "rmse", maximize = FALSE)
good_values <- select_best(grid_results, metric = "rmse", maximize = FALSE) # or manually select

# 5. finalise recipe and model
new_rec <- finalize_recipe(this_recipe, good_values)
new_mod <- finalize_model(this_model, good_values)

# 6. evaluate on test set


# 7. predict or infer, as required
