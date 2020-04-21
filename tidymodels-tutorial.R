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
dia_vfold %>% 
  mutate(df_ana = map(splits, analysis),
         df_ass = map(splits, assessment))

