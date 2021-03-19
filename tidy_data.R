
# tidy data ---------------------------------------------------------------

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

# load in dataset
covid_data <- 
  read_csv("data/unprocessed/reported_hospital_utilization_timeseries_20210227_1306.csv") %>% 
  clean_names()

# filter out "disqualified" predictors
covid_data <- 
  covid_data %>% 
  select(-ends_with(c("_numerator", "_denominator",
                      "_no", "_not_reported", "within_week_yes")))

# create correlation matrix
corr_matrix_tot <- covid_data %>% 
  # unselect non-numeric type variables
  select(-c(state, date)) %>% 
  # remove rows with missing data
  drop_na() %>%
  cor()

# correlations with response variable
# arrange in ascending order of correlation coefficient
corr_response <- corr_matrix_tot %>% 
  as_tibble() %>% 
  # select the column showing the correlation with the outcome var
  slice(1) %>% 
  # pivoting, make a character column of variables' names
  # and a numeric column of values of the correlation coefficients
  pivot_longer(everything(), 
               names_to = "variable", values_to = "correlation") %>%
  arrange(desc(correlation))

# show the correlation with the response variable in a scrollable box
kbl(corr_response) %>%
  kable_paper() %>%
  scroll_box(width = "100%", height = "200px")


# number of variables with relatively weak correlations with the outcome
corr_response %>% 
  filter(correlation >= 0.9) %>% 
  nrow()

# visualize the correlation of the response var against all others
covid_data %>% corr_var(
  # name of variable to focus on
  critical_staffing_shortage_today_yes, 
  top = 10
)

# correlations between predictors
corr_pred <- covid_data %>% 
  # unselect outcome variable
  select(-c(critical_staffing_shortage_today_yes)) %>% 
  # temporarily change `state` and `date` to type numeric
  mutate(date = as.numeric(date),
         # first turn state into a factor
         # then turn it into a numeric variable
         # with values determined by factor levels
         state = as.numeric(as.factor(state))) %>%
  # remove rows with missing data
  drop_na() %>%
  # correlation matrix
  correlate() %>% 
  # turn into a tibble
  stretch() %>% 
  rename("correlation" = "r") %>% 
  # remove rows with invalid result
  # the `NA` values are the correlation of one var with itself
  # in this case
  filter(!is.na(correlation)) %>% 
  # arranging in descending order
  arrange(desc(correlation))

# remove even rows
# they just repeat the info of the odd rows above them
corr_pred <- corr_pred %>% 
  # temporary var `row_id` to help the removing process
  mutate(row_id = row_number()) %>% 
  # filter out even rows
  filter(!row_id %% 2 == 0) %>% 
  # remove temporary var
  select(-row_id)

# show the correlations between predictors in a scrollable box
kbl(corr_pred) %>%
  kable_paper() %>%
  scroll_box(width = "100%", height = "200px")

# explore collinearity
near_perfect_collinearity <- corr_pred %>% 
  filter(abs(correlation) > 0.90)

near_perfect_collinearity %>% 
  distinct(x, y) %>% 
  kbl() %>%
  kable_paper() %>%
  scroll_box(width = "100%", height = "200px") %>%
  footnote(general = "Predictor pairs with near perfect collinearity")

# visualize relation btw different coverages
covid_data %>% 
  ggplot(aes(inpatient_beds_used_coverage, 
             inpatient_beds_utilization_coverage)) + 
  geom_point() + 
  geom_smooth(se = FALSE) + 
  labs(
    title = "Relation between Different Coverage Variables"
  )

# correlation plot between all numeric predictors
# vector for the temporary column names
temp_col_names <- 
  # indexed strings
  paste(c("X"), 1:21, sep="")

covid_data %>% 
  # select only numeric predictors
  select(ends_with("_coverage")) %>% 
  # temporarily rename all columns to indexed strings
  rename_all(~ temp_col_names) %>% 
  drop_na() %>% 
  # compute correlation matrix
  cor() %>% 
  # visualize
  corrplot(type = "upper", 
           title = "Correlations between Variables Ending with `_coverage`")

covid_data <- covid_data %>% 
  # remove variables with near perfect collinearity
  # only keep `inpatient_beds_coverage`
  select(-c(ends_with("_coverage")), inpatient_beds_coverage)

# remaining strong collinearity
remain_near_perfect_collinearity <- covid_data %>% 
  # unselect outcome variable
  select(-c(critical_staffing_shortage_today_yes)) %>% 
  # temporarily change `state` and `date` to type numeric
  mutate(date = as.numeric(date),
         # first turn state into a factor
         # then turn it into a numeric variable
         # with values determined by factor levels
         state = as.numeric(as.factor(state))) %>%
  # remove rows with missing data
  drop_na() %>%
  # correlation matrix
  correlate() %>% 
  # turn into a tibble
  stretch() %>% 
  rename("correlation" = "r") %>% 
  # obtain rows with correlation greater than 0.90
  filter(correlation > 0.9) %>% 
  arrange(desc(correlation))

# show tibble
kbl(remain_near_perfect_collinearity) %>%
  kable_paper() %>%
  scroll_box(width = "100%", height = "200px")

# visualize inter-variable correlations between predictors
covid_data %>% 
  ggplot(aes(previous_day_admission_adult_covid_confirmed, 
             total_adult_patients_hospitalized_confirmed_covid)) + 
  geom_point() + 
  labs(
    title =  "Relation Between Previous Day and Total Confirmed Cases", 
    x = "Previous Day Confirmed Cases", 
    y = "Total Confirmed Cases"
  )

# visualize inter-variable correlations between predictors
covid_data %>% 
  ggplot(aes(total_adult_patients_hospitalized_confirmed_covid, 
             total_adult_patients_hospitalized_confirmed_and_suspected_covid)) +
  geom_point() + 
  labs(
    title =  "Relation Between Confirmed and Confirmed and Suspected Cases", 
    x = "Confirmed Cases", 
    y = "Confirmed and Suspected Cases"
  )

# visualize inter-variable correlations between predictors
covid_data %>% 
  ggplot(aes(previous_day_admission_adult_covid_confirmed, 
             total_adult_patients_hospitalized_confirmed_covid)) +
  geom_point() + 
  labs(
    title =  "Relation Between Previous Day and Total Confirmed Cases", 
    x = "Previous Day Comfirmed Cases", 
    y = "Total Confirmed Cases"
  )

covid_data %>% 
  mutate(
    total_confirmed_and_suspected = 
      total_adult_patients_hospitalized_confirmed_and_suspected_covid + 
      total_pediatric_patients_hospitalized_confirmed_and_suspected_covid
  ) %>% 
  ggplot(aes(inpatient_beds_used_covid, 
             total_confirmed_and_suspected)) +
  geom_point() + 
  geom_smooth(se = FALSE) + 
  labs(
    title =  "Inpatient Bed Usage and Confirmed and Suspected Cases", 
    x = "Inpatient Bed Usage Related to COVID", 
    y = "Confirmed and Suspected Cases"
  )

# remove redundant predictors
covid_data <- covid_data %>% 
  select(-contains("_covid"), inpatient_beds_used_covid)

# visualize inter-variable correlations between predictors
covid_data %>% 
  ggplot(aes(inpatient_beds, 
             inpatient_beds_used)) +
  geom_point() + 
  geom_smooth(se = FALSE) + 
  labs(
    title =  "Relation Between Total and Used Inpatient Beds", 
    x = "Total Inpatient Beds", 
    y = "Used Inpatient Beds"
  )

covid_data %>% 
  ggplot(aes(inpatient_beds, 
             total_staffed_adult_icu_beds)) +
  geom_point() +
  labs(
    title =  "Relation Between Total Inpatient Beds and ICU Beds", 
    x = "Total Inpatient Beds", 
    y = "Staffed ICU Beds"
  )

# remove redundant predictors
covid_data <- covid_data %>% 
  select(-c(`inpatient_beds_used`, `staffed_adult_icu_bed_occupancy`, 
            `total_staffed_adult_icu_beds`))

# remaining predictors correlations
corr_pred_remain <- covid_data %>% 
  # unselect outcome variable
  select(-c(critical_staffing_shortage_today_yes)) %>% 
  # temporarily change `state` and `date` to type numeric
  mutate(date = as.numeric(date),
         # first turn state into a factor
         # then turn it into a numeric variable
         # with values determined by factor levels
         state = as.numeric(as.factor(state))) %>%
  # remove rows with missing data
  drop_na() %>%
  # correlation matrix
  correlate() %>% 
  # turn into a tibble
  stretch() %>% 
  rename("correlation" = "r") %>% 
  # arrange in descending order of correlation
  arrange(desc(correlation))

# show tibble
kbl(corr_pred_remain) %>%
  kable_paper() %>%
  scroll_box(width = "100%", height = "200px")

# check for correlation exceeding 0.9
corr_pred_remain %>% 
  filter(abs(correlation) > 0.9) %>% 
  nrow()

# correlation plot between predictors
covid_data %>% 
  # select only predictors
  select(-c(critical_staffing_shortage_today_yes)) %>% 
  # temporarily change `state` and `date` to type numeric
  mutate(date = as.numeric(date),
         # first turn state into a factor
         # then turn it into a numeric variable
         # with values determined by factor levels
         state = as.numeric(as.factor(state))) %>%
  # remove rows with missing data
  drop_na() %>% 
  # compute correlation matrix
  cor() %>% 
  # visualize
  corrplot(type = "upper",  tl.pos = "td",
           method = "circle", tl.cex = 0.5, tl.col = 'black',
           order = "hclust", diag = FALSE)

# overview of the missingness situation
covid_data %>% 
  vis_miss(cluster = TRUE)

# summary for missing data for predictors
pred_miss_summary <- covid_data %>% 
  select(-critical_staffing_shortage_today_yes) %>% 
  miss_var_summary()

# visualize the distribution of percentage of missing values 
pred_miss_summary %>% 
  ggplot(aes(pct_miss)) + 
  # add histogram
  geom_histogram() +
  labs(
    x = "Percent Missing",
    title = "Distribution of the Percentage of Missingness"
  )

# remove rows containing NA values
covid_data <- covid_data %>% 
  drop_na()

# visualize the distribution of valid observations overtime
covid_data %>% 
  ggplot(aes(date)) + 
  geom_histogram() + 
  labs(
    title = "Number of Observations at Each Date"
  )

# filter out dates
covid_data <- covid_data %>% 
  filter(date >= "2020-08-01")

# visualize distribution of `state`
covid_data %>% 
  ggplot(aes(y = state)) + 
  geom_bar() + 
  xlim(0, 250) + 
  labs(y = NULL)

# collapsing `state` levels to regions
covid_data <- covid_data %>% 
  mutate(state = 
           fct_collapse(state, "Northeast" = c("ME", "VT", "NH", "MA", 
                                               "NY", "RI", "CT", "NJ", "PA"), 
                        "Midwest" = c("ND", "SD", "MN", "WI", "MI", "IA", 
                                      "IL", "IN", "OH", "NE", "KS", "MO"), 
                        "Southeast" = c("MD", "DE", "WV", "VA", "KY", "TN", 
                                        "NC", "SC", "GA", "AL", "MS", "AR",
                                        "LA", "FL"), 
                        "Southwest" =c("AZ", "NM", "TX", "OK"), 
                        "West" = c("AK", "WA", "OR", "MT", "ID", 
                                   "WY", "CA", "NV", "UT", "CO"), 
                        "other" = c("PR", "VI", "DC", "HI"))) %>% 
  rename(state_region = state)

# visualize outcome var
covid_data %>% 
  ggplot(aes(critical_staffing_shortage_today_yes)) + 
  geom_histogram() + 
  labs(
    title = "Distribution of Response Variable", 
    subtitle = "critical_staffing_shortage_today_yes",
    x = "Number of Hospitals Reporting Critical Staff Shortage"
  )

# log-transform outcome var
covid_data <- covid_data %>% 
  mutate(
    critical_shortage_log = 
      log10(critical_staffing_shortage_today_yes + 0.00000001)
  ) %>% 
  select(-critical_staffing_shortage_today_yes)

# store processed data
covid_data %>% 
  write_rds("data/processed/covid_data.rds")
