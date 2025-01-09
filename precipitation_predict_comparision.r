#' following code presents application of different regression models
#' to predict precipitation basing on given dataset
#' and final comparision for models using RSME and R-squared

# importing necessary packages
install.packages("tidymodels")
install.packages("tidyverse")
install.packages("glmnet")

library(tidymodels)
library(tidyverse)
library(glmnet)

# download and unpacking data
url <- "https://dax-cdn.cdn.appdomain.cloud/dax-noaa-weather-data-jfk-airport/1.1.4/noaa-weather-sample-data.tar.gz"
download.file(url, destfile = "noaa_weather_sample.tar.gz")
untar("noaa_weather_sample.tar.gz", tar = "internal")
noaa_sample_df <- read.csv("noaa-weather-sample-data/jfk_weather_sample.csv")

# data filter and clean
sub_weather <- noaa_sample_df %>%
  select(
	HOURLYRelativeHumidity,
	HOURLYDRYBULBTEMPF,
	HOURLYPrecip,
	HOURLYWindSpeed,
	HOURLYStationPressure) %>%
# removing all non-numeric values from HOURLYPrecip column:
# T('trace amounts') converted to 0.0
# values ending with 's' (for snow) dropped to numeric
  mutate_at(
	'HOURLYPrecip', ~ str_remove(str_replace(., "T", "0.0"), pattern = "s$")
	) %>%
  drop_na(HOURLYPrecip) %>%
  mutate_if(is.character, as.numeric) %>%
  rename(relative_humidity = HOURLYRelativeHumidity,
         dry_bulb_temp_f = HOURLYDRYBULBTEMPF,
         precip = HOURLYPrecip,
         wind_speed = HOURLYWindSpeed,
         station_pressure = HOURLYStationPressure)

# splitting dataset in train and test parts
set.seed(1234)
weather_split <- initial_split(sub_weather, prop = 0.8)
train_data <- training(weather_split)
test_data <- testing(weather_split)

# pick linear model
lm_mod <- linear_reg() %>%
  set_engine(engine = "lm")

# linear regression using humidity as predictor for precipitation

# train a model
precip_humidity_fit <- lm_mod %>%
  fit(precip ~ relative_humidity, data = train_data)
# calculate train and test results
train_results_precip_humidity <- precip_humidity_fit %>%
  predict(new_data = train_data) %>%
  mutate(truth = train_data$precip)
test_results_precip_humidity <- precip_humidity_fit %>%
  predict(new_data = test_data) %>%
  mutate(truth = test_data$precip)
# root mean squared error for precipitation-humidity linear model
rmse_train_prec_hum <- rmse(train_results_precip_humidity,
							truth = truth,
							estimate = .pred)$.estimate
rmse_test_prec_hum <- rmse(test_results_precip_humidity,
						   truth = truth,
						   estimate = .pred)$.estimate
# R-squared for precipitation-humidity linear model
rsq_train_prec_hum <- rsq(train_results_precip_humidity,
						  truth = truth,
						  estimate = .pred)
rsq_test_prec_hum <- rsq(test_results_precip_humidity,
						 truth = truth,
						 estimate = .pred)

# linear regression using temperature as predictor for precipitation

# train a model
precip_temp_fit <- lm_mod %>%
  fit(precip ~ dry_bulb_temp_f, data = train_data)
# calculate train and test results
train_results_precip_temp <- precip_temp_fit %>%
  predict(new_data = train_data) %>%
  mutate(truth = train_data$precip)
test_results_precip_temp <- precip_temp_fit %>%
  predict(new_data = test_data) %>%
  mutate(truth = test_data$precip)
# root mean squared error for precipitation-temperature linear model
rmse_train_prec_temp <- rmse(train_results_precip_temp,
							 truth = truth,
							 estimate = .pred)$.estimate
rmse_test_prec_temp <- rmse(test_results_precip_temp,
							truth = truth,
                            estimate = .pred)$.estimate
# R-squared for precipitation-temperature linear model
rsq_train_prec_temp <- rsq(train_results_precip_temp,
						   truth = truth,
                           estimate = .pred)
rsq_test_prec_temp <- rsq(test_results_precip_temp,
						  truth = truth,
                          estimate = .pred)

# linear regression using wind speed as predictor for precipitation

# train a model
precip_wind_speed_fit <- lm_mod %>%
  fit(precip ~ wind_speed, data = train_data)
# calculate train and test results
train_results_precip_wind_speed <- precip_wind_speed_fit %>%
  predict(new_data = train_data) %>%
  mutate(truth = train_data$precip)
test_results_precip_wind_speed <- precip_wind_speed_fit %>%
  predict(new_data = test_data) %>%
  mutate(truth = test_data$precip)
# root mean squared error for precipitation-wind_speed linear model
rmse_train_prec_wind <- rmse(train_results_precip_wind_speed,
							 truth = truth,
                             estimate = .pred)$.estimate
rmse_test_prec_wind <- rmse(test_results_precip_wind_speed,
							truth = truth,
                            estimate = .pred)$.estimate
# R-squared for precipitation-wind_speed linear model
rsq_train_prec_wind <- rsq(train_results_precip_wind_speed,
						   truth = truth,
                           estimate = .pred)
rsq_test_prec_wind <- rsq(test_results_precip_wind_speed,
						  truth = truth,
                          estimate = .pred)

# linear regression using pressure as predictor for precipitation

# train a model
precip_pressure_fit <- lm_mod %>%
  fit(precip ~ station_pressure, data = train_data)
# calculate train and test results
train_results_precip_pressure <- precip_pressure_fit %>%
  predict(new_data = train_data) %>%
  mutate(truth = train_data$precip)
test_results_precip_pressure <- precip_pressure_fit %>%
  predict(new_data = test_data) %>%
  mutate(truth = test_data$precip)
# root mean squared error for precipitation-pressure linear model
rmse_train_prec_press <- rmse(train_results_precip_pressure,
							  truth = truth,
                              estimate = .pred)$.estimate
rmse_test_prec_press <- rmse(test_results_precip_pressure,
							 truth = truth,
                             estimate = .pred)$.estimate
# R-squared for precipitation-pressure linear model
rsq_train_prec_press <- rsq(train_results_precip_pressure,
							truth = truth,
                            estimate = .pred)
rsq_test_prec_press <- rsq(test_results_precip_pressure,
						   truth = truth,
                           estimate = .pred)


# developing improved models

# defining recipe for model formula to predict precipitation
precip_recipe <- recipe(precip ~ ., data = train_data)
# workflow combining pre-processing, modeling, and post-processing requests
precip_wf <- workflow() %>%
  add_recipe(precip_recipe)
# defining cross-validation to resample the data
precip_cvfolds <- vfold_cv(drop_na(train_data))
# setting up a grid to tune hyperparameters
lambda_grid <- grid_regular(levels = 50,
                            penalty(range = c(-3, 0.3)))

# lasso regularization
# defininig a model for penalty tuning
tune_lasso_model <- linear_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet")
lasso_grid <- tune_grid(
  precip_wf %>% add_model(tune_lasso_model), 
  resamples = precip_cvfolds, 
  grid = lambda_grid)

# returns best penalty value for lasso model
lasso_grid %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  slice(which.min(mean))

# training lasso model
lasso_precip_mod <- linear_reg(penalty = 0.00159, mixture = 1) %>%
  set_engine("glmnet")
lasso_fit <- precip_wf %>%
  add_model(lasso_precip_mod) %>%
  fit(data = drop_na(train_data))
lasso_fit %>%
  extract_fit_parsnip() %>%
  tidy()

# calculate train and test results
lasso_train_res <- lasso_fit %>%
  predict(new_data = drop_na(train_data)) %>%
  mutate(truth = drop_na(train_data)$precip)
lasso_test_res <- lasso_fit %>%
  predict(new_data = drop_na(test_data)) %>%
  mutate(truth = drop_na(test_data)$precip)
# root mean squared error for lasso model
rmse_train_lasso <- sqrt(
		mean((lasso_train_res$truth - lasso_train_res$.pred)^2)
	)
rmse_test_lasso <- sqrt(
		mean((lasso_test_res$truth - lasso_test_res$.pred)^2)
	)
# R-squared for lasso model
rsq_train_lasso <- rsq(lasso_test_res,
					   truth = truth,
					   estimate = .pred)
rsq_test_lasso <- rsq(lasso_train_res,
					  truth = truth,
					  estimate = .pred)

# ridge regularization
# defininig a model for penalty tuning
tune_ridge_model <- linear_reg(penalty = tune(), mixture = 0) %>% 
  set_engine("glmnet")
ridge_grid <- tune_grid(
  precip_wf %>% add_model(tune_ridge_model), 
  resamples = precip_cvfolds, 
  grid = lambda_grid)

# returns best penalty value for ridge model
ridge_grid %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  slice(which.min(mean))

# training ridge model
ridge_precip_mod <- linear_reg(penalty = 0.0140, mixture = 0) %>%
  set_engine("glmnet")
ridge_fit <- precip_wf %>%
  add_model(ridge_precip_mod) %>%
  fit(data = drop_na(train_data))
ridge_fit %>%
  extract_fit_parsnip() %>%
  tidy()

# calculate train and test results
ridge_train_res <- ridge_fit %>%
  predict(new_data = drop_na(train_data)) %>%
  mutate(truth = drop_na(train_data)$precip)
ridge_test_res <- ridge_fit %>%
  predict(new_data = drop_na(test_data)) %>%
  mutate(truth = drop_na(test_data)$precip)
# root mean squared error for ridge model
rmse_train_ridge <- sqrt(
		mean((ridge_train_res$truth - ridge_train_res$.pred)^2)
	)
rmse_test_ridge <- sqrt(
		mean((ridge_test_res$truth - ridge_test_res$.pred)^2)
	)
# R-squared for ridge model
rsq_train_ridge <- rsq(ridge_train_res,
					   truth = truth,
					   estimate = .pred)
rsq_test_ridge <- rsq(ridge_test_res,
					  truth = truth,
					  estimate = .pred)

# elastic net regularization
# defininig a model for penalty tuning
tune_elacticnet_mod <- linear_reg(penalty = tune(), mixture = 0.3) %>% 
  set_engine("glmnet")
elasticnet_grid <- tune_grid(
  precip_wf %>% add_model(tune_elacticnet_mod), 
  resamples = precip_cvfolds, 
  grid = lambda_grid)

# returns lowest penalty value for elastic net
elasticnet_grid %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  slice(which.min(mean))

# training elastic net model
elasticnet_mod <- linear_reg(penalty = 0.00404, mixture = 0.3) %>%
  set_engine("glmnet")
elasticnet_fit <- precip_wf %>%
  add_model(elasticnet_mod) %>%
  fit(data = drop_na(train_data))
elasticnet_fit %>%
  extract_fit_parsnip() %>%
  tidy()

# regularization models errors
elasticnet_train_res <- elasticnet_fit %>%
  predict(new_data = drop_na(train_data)) %>%
  mutate(truth = drop_na(train_data)$precip)
elasticnet_test_res <- elasticnet_fit %>%
  predict(new_data = drop_na(test_data)) %>%
  mutate(truth = drop_na(test_data)$precip)
# root mean squared error for elastic net model
rmse_train_elasticnet <- sqrt(
		mean((elasticnet_train_res$truth - elasticnet_train_res$.pred)^2)
	)
rmse_test_elasticnet <- sqrt(
		mean((elasticnet_test_res$truth - elasticnet_test_res$.pred)^2)
	)
# R-squared for elastic net model
rsq_train_elasticnet <- rsq(elasticnet_train_res,
					   truth = truth,
					   estimate = .pred)
rsq_test_elasticnet <- rsq(elasticnet_test_res,
					  truth = truth,
					  estimate = .pred)


# polynomial regression models

# 3rd order polynomial model using humidity as predictor for precipitation
precip_humidity_poly_fit <- lm_mod %>% 
  fit(precip ~ poly(relative_humidity, 3), 
      data = train_data)
# calculate train and test results
train_results_precip_humidity_poly <- precip_humidity_poly_fit %>%
  predict(new_data = train_data) %>%
  mutate(truth = train_data$precip)
test_results_precip_humidity_poly <- precip_humidity_poly_fit %>%
  predict(new_data = test_data) %>%
  mutate(truth = test_data$precip)
# root mean squared error for precipitation-humidity polynomial model
rmse_train_prec_hum_poly <- sqrt(
  mean(
    (train_results_precip_humidity_poly$truth - 
     train_results_precip_humidity_poly$.pred)^2
  ))
rmse_test_prec_hum_poly <- sqrt(
  mean(
    (test_results_precip_humidity_poly$truth - 
     test_results_precip_humidity_poly$.pred)^2
  ))
# R-squared for precipitation-humidity polynomial model
rsq_train_prec_hum_poly <- rsq(train_results_precip_humidity_poly,
							   truth = truth,
							   estimate = .pred)
rsq_test_prec_hum_poly <- rsq(test_results_precip_humidity_poly,
							  truth = truth,
							  estimate = .pred)

# 3rd order polynomial model using temperature as predictor for precipitation
precip_temp_poly_fit <- lm_mod %>% 
  fit(precip ~ poly(dry_bulb_temp_f, 3), 
      data = train_data)
# calculate train and test results
train_results_precip_temp_poly <- precip_temp_poly_fit %>%
  predict(new_data = train_data) %>%
  mutate(truth = train_data$precip)
test_results_precip_temp_poly <- precip_temp_poly_fit %>%
  predict(new_data = test_data) %>%
  mutate(truth = test_data$precip)
# root mean squared error for precipitation-temperature polynomial model
rmse_train_prec_temp_poly <- sqrt(
  mean(
    (train_results_precip_temp_poly$truth - 
     train_results_precip_temp_poly$.pred)^2
  ))
rmse_test_prec_temp_poly <- sqrt(
  mean(
    (test_results_precip_temp_poly$truth - 
     test_results_precip_temp_poly$.pred)^2
  ))
# R-squared for precipitation-temperature polynomial model
rsq_train_prec_temp_poly <- rsq(train_results_precip_temp_poly,
								truth = truth,
								estimate = .pred)
rsq_test_prec_temp_poly <- rsq(test_results_precip_temp_poly,
							   truth = truth,
							   estimate = .pred)

# 3rd order polynomial model using wind speed as predictor for precipitation
precip_wind_poly_fit <- lm_mod %>% 
  fit(precip ~  poly(wind_speed, 3), 
      data = train_data)
# calculate train and test results
train_results_precip_wind_poly <- precip_wind_poly_fit %>%
  predict(new_data = train_data) %>%
  mutate(truth = train_data$precip)
test_results_precip_wind_poly <- precip_wind_poly_fit %>%
  predict(new_data = test_data) %>%
  mutate(truth = test_data$precip)
# root mean squared error for precipitation-wind_speed polynomial model
rmse_train_prec_wind_poly <- sqrt(
  mean(
    (train_results_precip_wind_poly$truth - 
     train_results_precip_wind_poly$.pred)^2
  ))
rmse_test_prec_wind_poly <- sqrt(
  mean(
    (test_results_precip_wind_poly$truth - 
     test_results_precip_wind_poly$.pred)^2
  ))
# R-squared for precipitation-wind_speed polynomial model
rsq_train_prec_wind_poly <- rsq(train_results_precip_wind_poly,
								truth = truth,
								estimate = .pred)
rsq_test_prec_wind_poly <- rsq(test_results_precip_wind_poly,
							   truth = truth,
							   estimate = .pred)

# 3rd order polynomial model using pressure as predictor for precipitation
precip_pres_poly_fit <- lm_mod %>% 
  fit(precip ~  poly(station_pressure, 3), 
      data = drop_na(train_data))
# calculate train and test results
train_results_precip_pres_poly <- precip_pres_poly_fit %>%
  predict(new_data = train_data) %>%
  mutate(truth = train_data$precip)
test_results_precip_pres_poly <- precip_pres_poly_fit %>%
  predict(new_data = test_data) %>%
  mutate(truth = test_data$precip)
# root mean squared error for precipitation-pressure polynomial model
rmse_train_prec_pres_poly <- sqrt(
  mean(
    (train_results_precip_pres_poly$truth - 
     train_results_precip_pres_poly$.pred)^2
  ))
rmse_test_prec_pres_poly <- sqrt(
  mean(
    (test_results_precip_pres_poly$truth - 
     test_results_precip_pres_poly$.pred)^2
  ))
# R-squared for precipitation-pressure polynomial model
rsq_train_prec_pres_poly <- rsq(train_results_precip_pres_poly,
								truth = truth,
								estimate = .pred)
rsq_test_prec_pres_poly <- rsq(test_results_precip_pres_poly,
							   truth = truth,
							   estimate = .pred)


# comparing models
model_names <- c(
  "precipitation-humidity linear",
  "precipitation-temperature linear",
  "precipitation-wind_speed linear",
  "precipitation-pressure linear",
  "precipitation-humidity polynomial",
  "precipitation-temperature polynomial",
  "precipitation-wind_speed polynomial",
  "precipitation-pressure polynomial",
  "lasso regularization",
  "ridge regularization",
  "elastic net regularization"
)

train_rmse <- c(
  rmse_train_prec_hum,
  rmse_train_prec_temp,
  rmse_train_prec_wind,
  rmse_train_prec_press,
  rmse_train_prec_hum_poly,
  rmse_train_prec_temp_poly,
  rmse_train_prec_wind_poly,
  rmse_train_prec_pres_poly,
  rmse_train_lasso,
  rmse_train_ridge,
  rmse_train_elasticnet
)

test_rmse <- c(
  rmse_test_prec_hum,
  rmse_test_prec_temp,
  rmse_test_prec_wind,
  rmse_test_prec_press,
  rmse_test_prec_hum_poly,
  rmse_test_prec_temp_poly,
  rmse_test_prec_wind_poly,
  rmse_test_prec_pres_poly,
  rmse_test_lasso,
  rmse_test_ridge,
  rmse_test_elasticnet
)

# resulting dataframe to compare models basing on their RMSE
# regularized linear models show lowest error rate
# linear model, using humidity as predictor,
# can be treated as second best behind regularized
rmse_comparision_df <- data.frame(model_names, train_rmse, test_rmse)

train_rsq <- c(
  rsq_train_prec_hum$.estimate,
  rsq_train_prec_temp$.estimate,
  rsq_train_prec_wind$.estimate,
  rsq_train_prec_press$.estimate,
  rsq_train_prec_hum_poly$.estimate,
  rsq_train_prec_temp_poly$.estimate,
  rsq_train_prec_wind_poly$.estimate,
  rsq_train_prec_pres_poly$.estimate,
  rsq_train_lasso$.estimate,
  rsq_train_ridge$.estimate,
  rsq_train_elasticnet$.estimate
)

test_rsq <- c(
  rsq_test_prec_hum$.estimate,
  rsq_test_prec_temp$.estimate,
  rsq_test_prec_wind$.estimate,
  rsq_test_prec_press$.estimate,
  rsq_test_prec_hum_poly$.estimate,
  rsq_test_prec_temp_poly$.estimate,
  rsq_test_prec_wind_poly$.estimate,
  rsq_test_prec_pres_poly$.estimate,
  rsq_test_lasso$.estimate,
  rsq_test_ridge$.estimate,
  rsq_test_elasticnet$.estimate
)

# resulting dataframe to compare models basing on their R-squared
# regularized linear models perform a better correlation
# polynomial model, using pressure as predictor, also shows good results
rsq_comparision_df <- data.frame(model_names, train_rsq, test_rsq)