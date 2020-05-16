# from https://bgreenwell.github.io/pdp/articles/pdp-example-xgboost.html

# init --------------------------------------------------------------------

library(tidyverse)
library(xgboost)
library(pdp)
library(vip)

ames <- AmesHousing::make_ames()


# CV ----------------------------------------------------------------------

# Find the optimal number of rounds using 5-fold CV
set.seed(749)  # for reproducibility
ames_xgb_cv <- xgb.cv(
  data = data.matrix(subset(ames, select = -Sale_Price)),
  label = ames$Sale_Price,
  objective = "reg:linear",
  verbose = FALSE,
  nrounds = 1000,
  max_depth = 5,
  eta = 0.1, gamma = 0,
  nfold = 5,
  early_stopping_rounds = 30
)

ames_xgb_cv$best_iteration  # optimal number of trees

# fit ---------------------------------------------------------------------

# Fit an XGBoost model to the Boston housing data
set.seed(804)  # for reproducibility
ames_xgb <- xgboost::xgboost(
  data = data.matrix(subset(ames, select = -Sale_Price)),
  label = ames$Sale_Price,
  objective = "reg:linear",
  verbose = FALSE,
  nrounds = ames_xgb_cv$best_iteration,
  max_depth = 5,
  eta = 0.1, gamma = 0
)


# variable importance -----------------------------------------------------

vip(ames_xgb, num_features = 10)  # 10 is the default

# c-ICE curves and PDPs for Overall_Qual and Gr_Liv_Area
x <- data.matrix(subset(ames, select = -Sale_Price))  # training features
p1 <- partial(ames_xgb, pred.var = "Overall_Qual",
              ice = TRUE, center = TRUE, 
              plot = TRUE, rug = TRUE, alpha = 0.07,
              plot.engine = "ggplot2", 
              train = x)
p2 <- partial(ames_xgb, pred.var = "Gr_Liv_Area",
              ice = TRUE, center = TRUE, 
              plot = TRUE, rug = TRUE, alpha = 0.07,
              plot.engine = "ggplot2",
              train = x)
# look for heterogeneity in ICE curves, suggesting possible interactions

p3 <- partial(ames_xgb, pred.var = c("Overall_Qual", "Gr_Liv_Area"),
              plot = TRUE, chull = TRUE,
              plot.engine = "ggplot2",
              train = x)

grid.arrange(p1, p2, p3, ncol = 3)

# to do: repeat using tidymodels!
