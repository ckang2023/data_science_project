# NN tune -----------------------------------------------------------------

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

nn_model <- nearest_neighbor(mode = "regression", 
                             neighbors = tune()) %>% 
  set_engine("kknn")

# Nearest neighbors model
nn_params <- parameters(nn_model) %>% 
  update(neighbors = neighbors(range = c(1L, 25L)))
# store regular grid
nn_grid <- grid_regular(nn_params, levels = 5)

# Nearest neighbors model
nn_workflow <- workflow() %>% 
  add_model(nn_model) %>% 
  add_recipe(covid_recipe)

# Nearest neighbors model
nn_tuned <- nn_workflow %>% 
  tune_grid(covid_folds, grid = nn_grid)

write_rds(nn_tuned, "results/nn_tune.rds")