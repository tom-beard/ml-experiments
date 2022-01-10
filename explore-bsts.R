# Explore bsts package

library(tidyverse)
library(lubridate)
library(here)
library(bsts)

# examples based on https://www.unofficialgoogledatascience.com/2017/07/fitting-bayesian-structural-time-series.html

data(iclaims)
initial.claims # a zoo object

y <- initial.claims$iclaimsNSA # the series that we're modelling


# define & fit model ------------------------------------------------------

ss <- list() %>% 
  AddLocalLinearTrend(y) %>%
  AddSeasonal(y, nseasons = 52)

View(ss)

model1 <- bsts(y,
               state.specification = ss,
               niter = 1000) # takes ~35s on my laptop

# examine model results ---------------------------------------------------

model1 %>% summary()
names(model1)
model1 %>% View()

model1$sigma.obs %>% str()
model1$state.specification %>% View()

plot(model1) # same as plot(model1, "state")
plot(model1, "components")
plot(model1, "residuals")
plot(model1, "coefficients") # only if regressors are specified
plot(model1, "seasonal") # figure margins too large
plot(model1, "prediction.errors")


# predictions -------------------------------------------------------------

pred1 <- predict.bsts(model1, horizon = 12)
View(pred1)

plot(pred1, plot.original = 156)


