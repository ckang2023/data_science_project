
# Predictive Modeling -----------------------------------------------------

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

# train and tune models
# random forest model
rf_model <- rand_forest(mode = "regression",
                        min_n = tune(),
                        mtry = tune()) %>% 
  set_engine("ranger")

# boosted tree model
bt_model <- boost_tree(mode = "regression", 
                       mtry = tune(), 
                       min_n = tune(), 
                       learn_rate = tune()) %>% 
  set_engine("xgboost")

# Nearest neighbors model
nn_model <- nearest_neighbor(mode = "regression", 
                             neighbors = tune()) %>% 
  set_engine("kknn")

# elastic net regression model
elastic_net_reg_model <- linear_reg(penalty = tune(), 
                                    mixture = tune()) %>% 
  set_engine("glmnet")

# random forest model
rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(1, 11)))
# store regular grid
rf_grid <- grid_regular(rf_params, levels = 5)

# boosted tree model
bt_params <- parameters(bt_model) %>% 
  update(mtry = mtry(range = c(1, 11)), 
         learn_rate = learn_rate(range = c(-1, 0)))
# store regular grid
bt_grid <- grid_regular(bt_params, levels = 5)

# Nearest neighbors model
nn_params <- parameters(nn_model) %>% 
  update(neighbors = neighbors(range = c(1L, 25L)))
# store regular grid
nn_grid <- grid_regular(nn_params, levels = 5)

# elastic net regression model
elastic_net_params <- parameters(elastic_net_reg_model)
# store regular grid
elastic_net_grid <- grid_regular(elastic_net_params, levels = 5)

# random forest model
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(covid_recipe)

# boosted tree model
bt_workflow <- workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(covid_recipe)

# Nearest neighbors model
nn_workflow <- workflow() %>% 
  add_model(nn_model) %>% 
  add_recipe(covid_recipe)

# elastic net regression model
elastic_net_workflow <- workflow() %>% 
  add_model(elastic_net_reg_model) %>% 
  add_recipe(covid_recipe)

# read in results
rf_tuned <- read_rds("results/rf_tune.rds")
bt_tuned <- read_rds("results/bt_tune.rds")
nn_tuned <- read_rds("results/nn_tune.rds")
elastic_net_tuned <- read_rds("results/elastic_net_tune.rds")

# show rmse and select parameters based on best numerical performance
# random forest model
show_best(rf_tuned, metric = "rmse") %>% 
  select(-.config) %>%
  kbl() %>% 
  kable_styling(bootstrap_options = c("striped", "hover", "condensed")) %>% 
  row_spec(1, color = "white",
           background = "blue")

# boosted tree model
show_best(bt_tuned, metric = "rmse") %>% 
  select(-.config) %>%
  kbl() %>% 
  kable_styling(bootstrap_options = c("striped", "hover", "condensed")) %>% 
  row_spec(1, color = "white",
           background = "blue")

# nearest neighbors model
show_best(nn_tuned, metric = "rmse") %>% 
  select(-.config) %>%
  kbl() %>% 
  kable_styling(bootstrap_options = c("striped", "hover", "condensed")) %>% 
  row_spec(1, color = "white",
           background = "blue")

# elastic net regression model
show_best(elastic_net_tuned, metric = "rmse") %>% 
  select(-.config) %>%
  kbl() %>% 
  kable_styling(bootstrap_options = c("striped", "hover", "condensed")) %>% 
  row_spec(1, color = "white",
           background = "blue")

rf_workflow_tuned <- rf_workflow %>% 
  finalize_workflow(select_best(rf_tuned, metric = "rmse"))

rf_results <- fit(rf_workflow_tuned, covid_train)

# create metric set
covid_metric <- yardstick::metric_set(yardstick::rmse, yardstick::mae, 
                                      yardstick::rsq)

# show result of prediction on testing data
predict(rf_results, new_data = covid_test) %>% 
  # change the predicted value from log-scale to original scale
  mutate(
    new_pred = round(10 ^ .pred)
  ) %>% 
  bind_cols(covid_test %>% select(critical_shortage_log)) %>% 
  # transform from log-scale back to original scale
  mutate(truth_obs = round(10 ^ critical_shortage_log)) %>%
  covid_metric(truth = truth_obs, estimate = new_pred) %>% 
  kbl() %>% 
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"))
