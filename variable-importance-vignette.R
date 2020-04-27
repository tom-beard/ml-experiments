# from https://koalaverse.github.io/vip/articles/vip.html

library(tidymodels)

# simulate training data -------------------------------------------------------------

set.seed(101)  # for reproducibility
trn <- mlbench::mlbench.friedman1(500) %>% as.data.frame() %>% as_tibble()

# trees & tree ensembles --------------------------------------------------------------------

# Load required packages
library(xgboost)  # for fitting GBMs
library(ranger)   # for fitting random forests
library(rpart)    # for fitting CART-like decision trees

# Fit a single regression tree
tree <- rpart(y ~ ., data = trn)

# Fit a random forest
set.seed(101)
rfo <- ranger(y ~ ., data = trn, importance = "impurity")

# Fit a GBM
set.seed(102)
bst <- xgboost(
  data = data.matrix(subset(trn, select = -y)),
  label = trn$y, 
  objective = "reg:linear",
  nrounds = 100, 
  max_depth = 5, 
  eta = 0.3,
  verbose = 0  # suppress printing
)

# to do: use parsnip idiom for the above

(vi_tree <- tree$variable.importance)
barplot(vi_tree, horiz = TRUE, las = 1)

(vi_rfo <- rfo$variable.importance)
barplot(vi_rfo, horiz = TRUE, las = 1)

(vi_bst <- xgb.importance(model = bst))
xgb.ggplot.importance(vi_bst)


# vip versions ------------------------------------------------------------

library(vip)

tree %>% vi()
rfo %>% vi()
bst %>% vi()

p1 <- vip(tree)
p2 <- vip(rfo, aesthetics = list(fill = "green3"))
p3 <- vip(bst, aesthetics = list(fill = "purple"))

grid.arrange(p1, p2, p3, ncol = 3)

bst %>% 
  vip(num_features = 5, geom = "point", horizontal = FALSE, 
      aesthetics = list(color = "red", shape = 17, size = 4)) +
  theme_light()
