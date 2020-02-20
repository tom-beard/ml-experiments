library(workflows)
library(ggplot2)
library(dplyr)
library(tidyr)
library(parsnip)
library(yardstick)
library(recipes)
library(discrim)

data("bivariate")

ggplot(bivariate_train, aes(x = A, y = B, col = Class)) + 
  geom_point(alpha = .3) + 
  coord_equal(ratio = 20)


bivariate_train %>% 
  pivot_longer(cols = c(A, B), names_to = "predictor") %>% 
  ggplot(aes(x = Class, y = value)) + 
  geom_boxplot() + 
  facet_wrap(~ predictor, scales = "free_y") + 
  scale_y_log10()

logit_model <- 
  logistic_reg() %>% 
  set_engine("glm")

glm_workflow <- 
  workflow() %>% 
  add_model(logit_model)

simple_glm <- 
  glm_workflow %>% 
  add_formula(Class ~ .) %>% 
  fit(data = bivariate_train)

simple_glm_probs <- 
  predict(simple_glm, bivariate_val, type = "prob") %>% 
  bind_cols(bivariate_val)

simple_glm_roc <- 
  simple_glm_probs %>% 
  roc_curve(Class, .pred_One)

simple_glm_probs %>% roc_auc(Class, .pred_One)
simple_glm_roc %>% autoplot()

ratio_glm <- 
  glm_workflow %>% 
  add_formula(Class ~ I(A/B)) %>% 
  fit(data = bivariate_train)

ratio_glm_probs <- 
  predict(ratio_glm, bivariate_val, type = "prob") %>% 
  bind_cols(bivariate_val)

ratio_glm_roc <- 
  ratio_glm_probs %>% 
  roc_curve(Class, .pred_One)

ratio_glm_probs %>% roc_auc(Class, .pred_One)
simple_glm_roc %>%
  autoplot() +
  geom_path(data = ratio_glm_roc, aes(x = 1- specificity, y = sensitivity), colour = "red")

transform_recipe <- 
  recipe(Class ~ ., data = bivariate_train) %>% 
  step_BoxCox(all_predictors())

updated_recipe <- prep(transform_recipe)
juice(updated_recipe)

transform_glm <- 
  glm_workflow %>% 
  add_recipe(transform_recipe) %>% 
  fit(data = bivariate_train)

transform_glm_probs <- 
  predict(transform_glm, bivariate_val, type = "prob") %>% 
  bind_cols(bivariate_val)

transform_glm_roc <- 
  transform_glm_probs %>% 
  roc_curve(Class, .pred_One)

transform_glm_probs %>% roc_auc(Class, .pred_One)
simple_glm_roc %>%
  autoplot() +
  geom_path(data = ratio_glm_roc, aes(x = 1 - specificity, y = sensitivity), colour = "red") +
  geom_path(data = transform_glm_roc, aes(x = 1 - specificity, y = sensitivity), colour = "blue")

ggplot(bivariate_train, aes(x = 1/A, y = 1/B, colour = Class)) +
  geom_point(alpha = 0.3) +
  coord_equal(ratio = 1/12)

pca_recipe <- 
  transform_recipe %>% 
  step_normalize(A, B) %>% 
  step_pca(A, B, num_comp = 2)
  
pca_glm <- 
  glm_workflow %>% 
  add_recipe(pca_recipe) %>% 
  fit(data = bivariate_train)

pca_glm_probs <- 
  predict(pca_glm, bivariate_val, type = "prob") %>% 
  bind_cols(bivariate_val)

pca_glm_roc <- 
  pca_glm_probs %>% 
  roc_curve(Class, .pred_One)

pca_glm_probs %>% roc_auc(Class, .pred_One)

discrim_model <- 
  discrim_flexible() %>% 
  set_engine("earth") %>% 
  set_mode("classification")

discrim_workflow <- 
  workflow() %>% 
  add_recipe(transform_recipe) %>% 
  add_model(discrim_model) %>% 
  fit(data = bivariate_train)

discrim_probs <- 
  predict(discrim_workflow, bivariate_val, type = "prob") %>% 
  bind_cols(bivariate_val)

discrim_roc <- 
  discrim_probs %>% 
  roc_curve(Class, .pred_One)

discrim_probs %>% roc_auc(Class, .pred_One)


# test set ----------------------------------------------------------------

test_probs <- 
  predict(transform_glm, bivariate_test, type = "prob") %>% 
  bind_cols(bivariate_test)

test_roc <- 
  test_probs %>% 
  roc_curve(Class, .pred_One)

test_probs %>% roc_auc(Class, .pred_One)
autoplot(test_roc)


# rationalise to "typical" workflow ---------------------------------------

# use lists to store models and workflows?


