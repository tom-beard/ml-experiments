library(tidymodels)

chi_rec <- recipe(ridership ~ ., data = Chicago) %>% 
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

chi_rec <- recipe(ridership ~ ., data = Chicago) %>% 
  step_holiday(date) %>% 
  step_date(date) %>% 
  step_rm(date) %>% 
  step_dummy(all_nominal()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(one_of(!!stations)) %>% 
  step_pca(one_of(!!stations), num_comp = tune()) %>% 
  step_normalize(all_predictors())

time_resamples <- rolling_origin(Chicago,
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
