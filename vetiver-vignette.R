# from https://rstudio.github.io/vetiver-r/articles/vetiver.html
N
library(parsnip)
library(recipes)
library(workflows)

data(bivariate, package = "modeldata")
bivariate_train

set.seed(1234)

biv_recipe <- 
  recipe(Class ~ ., data = bivariate_train) %>% 
  step_BoxCox(all_predictors()) %>% 
  step_normalize(all_predictors())

svm_spec <- 
  svm_linear(mode = "classification") %>% 
  set_engine("LiblineaR")

svm_fit <- 
  workflow(biv_recipe, svm_spec) %>% 
  fit(sample_frac(bivariate_train, 0.7))

library(vetiver)

v <- vetiver_model(svm_fit, "biv_svm")

library(pins)

model_board <- board_temp(versioned = TRUE)

# first version:
model_board %>% vetiver_pin_write(v)

# second version:
svm_fit <- 
  workflow(biv_recipe, svm_spec) %>% 
  fit(sample_frac(bivariate_train, 0.7))

v <- vetiver_model(svm_fit, "biv_svm")

model_board %>% vetiver_pin_write(v)

model_board
model_board %>% pin_versions("biv_svm")

# deploy via Plumber

library(plumber)

pr() %>% 
  vetiver_api(v)

# might be better to run the API in a background process

pr() %>% 
  vetiver_api(v) %>% 
  pr_run()

# vetiver_write_plumber(model_board, "biv_svm") # writes to working directory

endpoint <- vetiver_endpoint("http://127.0.0.1:8088/predict")
endpoint

data(bivariate, package = "modeldata")
predict(endpoint, bivariate_test)

# vetiver_write_docker(v) # writes to working directory
