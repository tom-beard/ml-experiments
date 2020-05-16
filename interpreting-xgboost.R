# from https://bgreenwell.github.io/pdp/articles/pdp-example-xgboost.html

# init --------------------------------------------------------------------

library(tidyverse)
library(xgboost)
library(pdp)
library(vip)

ames <- AmesHousing::make_ames()
