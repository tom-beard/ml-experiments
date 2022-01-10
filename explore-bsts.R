# Explore bsts package

library(tidyverse)
library(lubridate)
library(here)
library(bsts)

# examples based on https://www.unofficialgoogledatascience.com/2017/07/fitting-bayesian-structural-time-series.html

data(iclaims)
initial.claims # a zoo object

y <- initial.claims$iclaimsNSA # the series that we're modelling; it's already been transformed(?!)


# define & fit model ------------------------------------------------------

ss <- list() %>% 
  AddLocalLinearTrend(y) %>%
  AddSeasonal(y, nseasons = 52)

View(ss)

model1 <- bsts(y,
               state.specification = ss,
               niter = 1000) # takes ~35s on my laptop
# if you pass a full zoo or ts object, the timeseries info can be used in plotting etc

# examine model results ---------------------------------------------------

model1 %>% summary()
names(model1)
model1 %>% View()

model1$sigma.obs %>% str()
model1$state.specification %>% View()

plot(model1) # same as plot(model1, "state")
?plot(model1, "components")
plot(model1, "residuals")
plot(model1, "coefficients") # only if regressors are specified
plot(model1, "seasonal") # figure margins too large
plot(model1, "prediction.errors")


# extracting and visualising results --------------------------------------

# based on https://michaeldewittjr.com/programming/2018-07-05-bayesian-time-series-analysis-with-bsts_files/
# this could be made much more tidyverse-y!

length(model1$state.specification) # 2 components
comp_1_name <- model1$state.specification[[1]]$name
comp_2_name <- model1$state.specification[[2]]$name

times <- as.Date(model1$timestamp.info$timestamps)

components_df <- bind_rows(
  tibble(component = comp_1_name, date = times, value = colMeans(model1$state.contributions[, comp_1_name, ])),
  tibble(component = comp_2_name, date = times, value = colMeans(model1$state.contributions[, comp_2_name, ]))
)

components_df %>% 
  ggplot(aes(x = date, y = value)) +
  geom_line() + 
  theme_bw() +
  ylab("") + xlab("") + 
  facet_grid(rows = vars(component), scales = "free") +
  theme(legend.title = element_blank()) +
  theme(axis.text.x = element_text(angle = -90, hjust = 0))

# predictions -------------------------------------------------------------

pred1 <- predict.bsts(model1, horizon = 12)
View(pred1)

plot(pred1, plot.original = 156)


# test with zeroes and NAs ------------------------------------------------

y_2 <- pmax(initial.claims$iclaimsNSA, 0)
y_2[300:320] <- NA

plot(y_2)

ss_2 <- list() %>% 
  AddLocalLinearTrend(y_2) %>%
  AddSeasonal(y_2, nseasons = 52)

View(ss_2)

model_2 <- bsts(y_2,
               state.specification = ss_2,
               niter = 1000) # takes ~35s on my laptop
View(model_2)

plot(model_2)
plot(model_2, show.actuals = TRUE)
plot(model_2, "components")

# this produces results, but has a somewhat "forced" trend where there are zeroes


