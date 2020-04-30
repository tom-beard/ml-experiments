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


# linear models and MARS -----------------------------------------------------------

linmod <- lm(y ~ .^2, data = trn)
backward <- stats::step(linmod, direction = "backward", trace = 0) # recipes::step() conflicts with stats::step()

vi(backward)

backward %>% 
  vip(num_features = length(coef(backward)),
      geom = "point", horizontal = FALSE, mapping = aes_string(colour = "Sign"))

backward %>% 
  vi() %>% 
  separate(Variable, into = c("var1", "var2"), sep = ":", fill = "right") %>% 
  mutate(var2 = coalesce(var2, var1)) %>% 
  ggplot() +
  geom_point(aes(x = var1, y = var2, size = Importance, colour = Sign)) +
  labs(x = "", y = "", title = "") +
  theme_minimal() +
  theme(panel.grid.minor = element_blank())

library(earth)  

mars <- earth(y ~ ., data = trn, degree = 2, pmethod = "exhaustive")
vi(mars)
vip(mars)


# nn ----------------------------------------------------------------------

library(nnet)
set.seed(0803)
nn <- nnet(y ~ ., data = trn, size = 7, decay = 0.1, linout = TRUE)
grid.arrange(
  vip(nn),
  vip(nn, type = "garson"),
  nrow = 1
)

methods(vi_model) %>% as.character() %>% stringr::str_remove(., "^vi_model\\.")


# model-agnostic scores ---------------------------------------------------

library(pdp)

pp <- ppr(y ~ ., data = trn, nterms = 11)

features <- paste0("x.", 1:10)

pdps <- features %>% 
  map(~ partial(pp, pred.var = .x) %>%
               autoplot() +
               ylim(range(trn$y)) +
               theme_light()
      )
grid.arrange(grobs = pdps, ncol = 5)

# method = "firm" seems to have replaced method = "pdp" in vi
# p1 <- vip(pp, method = "pdp") + ggtitle("PPR")
# p2 <- vip(nn, method = "pdp") + ggtitle("NN")
p1 <- vip(pp, method = "firm") + ggtitle("PPR")
p2 <- vip(nn, method = "firm") + ggtitle("NN")
grid.arrange(p1, p2, ncol = 2)

ice_curves <- features %>% 
  map(~ partial(pp, pred.var = .x, ice = TRUE) %>%
        autoplot(alpha = 0.1) +
        ylim(range(trn$y)) +
        theme_light()
      )
grid.arrange(grobs = ice_curves, ncol = 5)

p1 <- vip(pp, method = "firm", ice = TRUE) + ggtitle("PPR")
p2 <- vip(nn, method = "firm", ice = TRUE) + ggtitle("NN")
grid.arrange(p1, p2, ncol = 2)

set.seed(2021)

p1 <- vip(pp, method = "permute", target = "y", metric = "rsquared", pred_wrapper = predict) + 
  ggtitle("PPR")
p2 <- vip(nn, method = "permute", target = "y", metric = "rsquared", pred_wrapper = predict) + 
  ggtitle("NN")
grid.arrange(p1, p2, ncol = 2)

set.seed(2021)
vip(pp, method = "permute", target = "y", metric = "rsquared", nsim = 20,
    pred_wrapper = predict,
    geom = "boxplot", all_permutations = TRUE,
    mapping = aes_string(fill = "Variable"),
    aesthetics = list(colour = "grey25")) + 
  ggtitle("PPR")


# diabetes data -----------------------------------------------------------

library(ranger)
data(pima, package = "pdp")
pima <- na.omit(pima) %>% as_tibble()

set.seed(1322)

rfo1 <- ranger(diabetes ~ ., data = pima, importance = "permutation")
rfo2 <- ranger(diabetes ~ ., data = pima, importance = "permutation", probability = TRUE)

p1 <- vip(rfo1) # model-specific
p2 <- vip(rfo2) # model-specific
set.seed(1329)
pfun <- function(object, newdata) predict(object, data = newdata)$predictions
p3 <- vip(rfo1, method = "permute", metric = "error", pred_wrapper = pfun,
          target = "diabetes")
p4 <- vip(rfo2, method = "permute", metric = "auc", pred_wrapper = pfun,
          target = "diabetes", reference_class = "neg")
grid.arrange(p1, p2, p3, p4, ncol = 2)

# why is importance nearly 0 for all vars in p4?

# sparklines --------------------------------------------------------------

var_imp <- vi(rfo2, method = "permute", metric = "auc", pred_wrapper = pfun,
              target = "diabetes", reference_class = "neg")
add_sparklines(var_imp, fit = rfo2)

nn %>% 
  vi(method = "firm") %>% 
  add_sparklines()

nn %>% 
  vi(method = "firm") %>% 
  add_sparklines(standardize_y = FALSE)


# interaction effects -----------------------------------------------------

# from https://koalaverse.github.io/vip/articles/vip-interaction.html

# Load required packages
# library(gbm)             # for fitting generalized boosted models
library(lattice)         # for general visualization
library(mlbench)         # for machine learning benchmark data sets
library(NeuralNetTools)  # for various tools for neural networks
# library(nnet)            # for fitting neural networks w/ a single hidden layer

set.seed(101)
trn <- mlbench::mlbench.friedman1(500, sd = 1) %>% as.data.frame() %>% as_tibble()

set.seed(103)
nn <- nnet(y ~ ., data = trn,
           size = 8, linout = TRUE, decay = 0.01, maxit = 1000, trace = FALSE)

p1 <- vip(nn, method = "firm", feature_names = paste0("x.", 1:10)) +
  labs(x = "", y = "Importance", title = "PDP method") +
  theme_light()

nn_garson <- garson(nn, bar_plot = FALSE) %>% 
  rownames_to_column("Variable") %>% 
  select(Variable, Importance = rel_imp)
p2 <- ggplot(nn_garson, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col() +
  labs(x = "", y = "Importance", title = "Garson's method") +
    coord_flip() +
  theme_light()

nn_olden <- olden(nn, bar_plot = FALSE) %>% 
  rownames_to_column("Variable") %>% 
  select(Variable, Importance = importance)
p3 <- ggplot(nn_olden, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col() +
  labs(x = "", y = "Importance", title = "Olden's method") +
    coord_flip() +
  theme_light()

grid.arrange(p1, p2, p3, ncol = 3)
