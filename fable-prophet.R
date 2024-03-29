# intro -------------------------------------------------------------------

# based on https://otexts.com/fpp3/prophet.html
# though that page doesn't specify packages to load

library(tidyverse)
library(fable)
library(fable.prophet)
library(tsibbledata)
library(lubridate)

cement <- aus_production %>%
  filter(year(Quarter) >= 1988)

train <- cement %>%
  filter(year(Quarter) <= 2007)

fit <- train %>%
  model(
    arima = ARIMA(Cement),
    ets = ETS(Cement),
    prophet = prophet(Cement ~ season(period = 4, order = 2,
                                      type = "multiplicative"))
  )

fc <- fit %>% forecast(h = "2 years 6 months")
fc %>% autoplot(cement)
fc %>% accuracy(cement)


# CRAN vignette -----------------------------------------------------------

# based on https://cran.r-project.org/web/packages/fable.prophet/vignettes/intro.html,
# but adapted to the Cement example

# components() creates a "dable" object (Decomposition table?)

fit %>%
  select(-arima) %>% # no components() method for ARIMA models
  components() %>% 
  autoplot() # ets doesn't throw an error, but is overplotted

fit %>%
  select(ets) %>%
  components() %>% 
  autoplot()

fit %>%
  select(prophet) %>%
  components() %>% 
  autoplot()


# get fitted values -------------------------------------------------------

fit %>%
  select(prophet) %>%
  fitted()
