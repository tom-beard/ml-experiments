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
