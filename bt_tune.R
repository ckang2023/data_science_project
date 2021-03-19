# Boosted Tree tune -----------------------------------------------------------------

# Load Packages
library(tidyverse)
library(janitor)
library(skimr)
library(tidymodels)
library(corrplot)
library(lubridate)
library(lares)
library(corrr)
library(kableExtra)
library(naniar)
library(embed)
library(xgboost)
library(kknn)

# set seed
set.seed(42)

# load data
covid_data <- 
  read_rds("data/processed/covid_data.rds")

# split data
covid_data <- initial_time_split(covid_data, prop = 0.7, 
                                 strata = critical_shortage_log)
# obtain training and test sets
covid_train <- training(covid_data)
covid_test <- testing(covid_data)

covid_folds <- 
  vfold_cv(data = covid_train, v = 10, repeats = 5)

covid_recipe <- 
  recipe(critical_shortage_log ~ ., 
         data = covid_train) %>% 
  # log-transform all numeric predictors
  step_log(c(inpatient_beds, inpatient_beds_coverage, 
             inpatient_beds_used_covid), offset = 0.0000001) %>% 
  step_other(state_region, threshold = 1000, 
             other = "Southwest_and_other") %>% 
  step_date(date, features = "doy") %>% 
  step_rm(date) %>%
  step_dummy(all_nominal(), one_hot = TRUE) %>% 
  # center and scale all predictors
  step_center(all_predictors()) %>% 
  step_scale(all_predictors())

bt_model <- boost_tree(mode = "regression", 
                       mtry = tune(), 
                       min_n = tune(), 
                       learn_rate = tune()) %>% 
  set_engine("xgboost")

# boosted tree model
bt_params <- parameters(bt_model) %>% 
  update(mtry = mtry(range = c(1, 11)), 
         learn_rate = learn_rate(range = c(-1, 0)))
# store regular grid
bt_grid <- grid_regular(bt_params, levels = 5)

# boosted tree model
bt_workflow <- workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(covid_recipe)

# boosted tree model
bt_tuned <- bt_workflow %>% 
  tune_grid(covid_folds, grid = bt_grid)

write_rds(bt_tuned, "results/bt_tune.rds")