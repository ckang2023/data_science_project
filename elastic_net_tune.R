# Elastic Net tune -----------------------------------------------------------------

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

elastic_net_reg_model <- linear_reg(penalty = tune(), 
                                    mixture = tune()) %>% 
  set_engine("glmnet")

# elastic net regression model
elastic_net_params <- parameters(elastic_net_reg_model)
# store regular grid
elastic_net_grid <- grid_regular(elastic_net_params, levels = 5)

# elastic net regression model
elastic_net_workflow <- workflow() %>% 
  add_model(elastic_net_reg_model) %>% 
  add_recipe(covid_recipe)

# elastic net regression model
elastic_net_tuned <- elastic_net_workflow %>% 
  tune_grid(covid_folds, grid = elastic_net_grid)

write_rds(elastic_net_tuned, "results/elastic_net_tune.rds")
