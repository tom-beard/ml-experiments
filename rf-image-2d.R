# init --------

library(tidyverse)
library(tidymodels)
library(doFuture)
library(ambient)
library(gridExtra)

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
folds <- 2

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
                       param_info = rf_param,
                       control = control_grid(extract = function (x) extract_model(x)))
beepr::beep()

# visualise all fits from grid ----------------------------------------------

test_extract <- rf_search$.extracts[[1]]

test_extract[1, ".extracts"][[1]][[1]] %>% class()
test_fit <- test_extract[1, ".extracts"][[1]][[1]]

test_preds <- predict(test_fit, data = data_cleaned)$predictions

predicted <- data_cleaned %>% 
  add_column(.pred = test_preds) %>% 
  mutate(residual = lum - .pred)

plot_pred <- predicted %>% 
  ggplot() +
  geom_tile(aes(x = x, y = y, fill = .pred)) +
  scale_y_reverse() +
  coord_equal()

plot_residual <- predicted %>% 
  ggplot() +
  geom_tile(aes(x = x, y = y, fill = residual)) +
  scale_fill_gradient2() +
  scale_y_reverse() +
  coord_equal()

grid.arrange(plot_pred, plot_residual, nrow = 1)
