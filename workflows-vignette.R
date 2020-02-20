library(workflows)
library(ggplot2)
library(dplyr)

data("bivariate")

ggplot(bivariate_train, aes(x = A, y = B, col = Class)) + 
  geom_point(alpha = .3) + 
  coord_equal(ratio = 20)
