# init --------

# from https://hansjoerg.me/2020/02/09/tidymodels-for-machine-learning/

library(tidymodels)
library(ggrepel)
library(corrplot)

theme_set(theme_light())


# data set: diamonds ------------------------------------------------------

data("diamonds")
diamonds %>% 
  sample_n(2000) %>% 
  mutate_if(is.factor, as.numeric) %>% 
  select(price, everything()) %>% 
  cor() %>% 
  {.[order(abs(.[, 1]), decreasing = TRUE),
     order(abs(.[, 1]), decreasing = TRUE)]} %>% 
  corrplot(method = "number", type = "upper", mar = c(0, 0, 1.5, 0),
           title = "Correlations between price and various features of diamonds")

# try using https://tidymodels.github.io/corrr/ instead?


# train/test split --------------------------------------------------------

set.seed(1243)

test_prop <- 0.1
folds <- 3

dia_split <- initial_split(diamonds, prop = test_prop, strata = price)
dia_train <- training(dia_split)
dia_test <- testing(dia_split)

dia_vfold <- vfold_cv(dia_train, v = folds, repeats = 1, strata = price)


# feature engineering -----------------------------------------------------

dia_train %>% 
  ggplot(aes(x = carat, y = price)) +
  geom_point(alpha = 0.5) +
  # geom_smooth(colour = "firebrick", fill = "red") +
  geom_smooth(method = "lm", formula = "y ~ poly(x, 4)") +
  scale_y_continuous(trans = log_trans(), labels = function (x) round(x, -2)) +
  labs(title = "Nonlinear relationship between price and carat of diamonds",
       subtitle = "The degree of the polynomial is a potential tuning parameter") +
  theme_minimal() +
  theme(panel.grid.minor = element_blank())
  
dia_recipe <- 
  recipe(price ~ ., data = dia_train) %>% 
  step_log(all_outcomes()) %>% 
  step_normalize(all_predictors(), -all_nominal()) %>% 
  step_dummy(all_nominal()) %>% 
  step_poly(carat, degree = 2)
  
dia_juiced <- dia_recipe %>% prep() %>% juice()


# defining and fitting models ---------------------------------------------

lm_model <- 
  linear_reg() %>% 
  set_mode("regression") %>% 
  set_engine("lm")
  
lm_fit1 <- lm_model %>% 
  fit(price ~ ., dia_juiced)

lm_fit1$fit %>% glance()
lm_fit1 %>%
  tidy() %>% 
  arrange(abs(p.value))

lm_predicted <- lm_fit1$fit %>% 
  augment(data = dia_juiced) %>% 
  rowid_to_column()

lm_predicted %>% 
  select(rowid, price, .fitted:.std.resid)

lm_predicted %>% 
  ggplot(aes(x = .fitted, y = price)) +
  geom_point(alpha = 0.2) +
  ggrepel::geom_label_repel(aes(label = rowid),
                            data = filter(lm_predicted, abs(.resid) > 1)) +
  labs(title = "Actual vs predicted price") +
  theme_minimal() +
  theme(panel.grid.minor = element_blank())


# evaluating model performance --------------------------------------------

# Note that the cross-validation code is a bit lengthy—for didactic
# purposes. Wrapper functions (e.g., from the tune package) would lead to more
# concise code: fit_resamples() could be used here (without tuning), or
# tune_grid() and tune_bayes().

lm_fit2 <- dia_vfold %>% 
  mutate(df_ana = map(splits, analysis),
         df_ass = map(splits, assessment)) %>% 
  mutate(recipe = map(df_ana, ~prep(dia_recipe, training = .x)),
         df_ana = map(recipe, juice),
         df_ass = map2(recipe, df_ass,
                       ~bake(.x, new_data = .y))) %>% 
  mutate(model_fit = map(df_ana,
                         ~fit(lm_model, price ~ ., data = .x))) %>% 
  mutate(model_pred = map2(model_fit, df_ass,
                           ~predict(.x, new_data = .y)))

lm_preds <- lm_fit2 %>% 
  mutate(res = map2(df_ass, model_pred, ~tibble(price = .x$price, .pred = .y$.pred))) %>% 
  select(id, res) %>% 
  unnest(res) %>% 
  group_by(id)

lm_preds %>% metrics(truth = price, estimate = .pred)
