# LGBM-Trip-Duration-Prediction-ML

# Taxi Trip Duration Prediction
## Project Description
The goal of this project is to develop accurate and robust predictive models for estimating taxi trip durations. This has real-world applications in the taxi and transportation industry.
## Problem Statement
Given a dataset of taxi trip records, including information such as pickup date and time, geographical coordinates (longitude and latitude) of the pickup and dropoff locations, the number of passengers, and other attributes, the task is to develop a regression model that can accurately predict the duration of a taxi trip in seconds. The model should be trained on a labeled dataset containing historical trip records (the training set) and then applied to a separate dataset of unlabeled trip records (the testing set) to estimate the trip durations.
## Dataset Description
The competition dataset is based on the 2016 NYC Yellow Cab trip record data made available in Big Query on Google Cloud Platform. The data was originally published by the NYC Taxi and Limousine Commission (TLC). The data was sampled and cleaned for the purposes of this playground competition. Participants should predict the duration of each trip in the test set based on individual trip attributes.

## Stages in the project
  ## 1. Data Loading – Loaded from Google Drive and connected to Colab Notebook.
  ## 2.EDA – EDA was done to understand the data and analyzed the data to gain insights.
  
![day of week](https://github.com/aiotsir/LGBM-Trip-Duration-Prediction-ML/assets/56543279/16b88c1d-2439-491f-bfc9-0e4f2237ffa6)

![hr of day and count](https://github.com/aiotsir/LGBM-Trip-Duration-Prediction-ML/assets/56543279/1e5fae4b-ca1c-4d9c-819c-d2d479d0b7b7)

  
      ## Insights from Data Analysis
                 Towards the end of the week (i.e., Thursday, Friday and Saturday), trip count is huge and is more than 2L.
                 Trips are the highest during 5PM and 6PM and generally, trips are more from 5PM to 10 PM.
                 Trips are least from 12 AM to 5AM.
                 Generally, trips are more from 8AM to 11 PM.


3.	## Data Cleaning and Visualization
4.	## Feature Engineering

## i.	Extracted relevant features from the data that can affect trip duration, such as:
1.	Distance between start and end points.
2.	Day of the week, time of day.
## ii.	Distance between start point and end point (lat_long) were calculated using the Haversine formula.  
•	The haversine function calculates the distance between two sets of coordinates. Here to calculate the distance between the pickup and drop-off point.

## iii.	Outlier Removal: There were outliers in trip_duration column, distance column, and passenger_count column.
1.	 Passenger count above 6 and below zero were removed. 
2.	distance above 600km were rare and distance below 100m doesn’t make much sense in taxi trip perspective.
3.	Used Z score method for removing outliers in the trip_duration column. The IQR method was not fruitful.
   
## iv.	Checked the data distribution in trip_duration column and found it to be left skewed. Therefore converted trip_duration to corresponding logrithmic values.
## v.	Correlation Analysis was done and plotted correlation heatmap.
## vi.	Checked variance of each feature.

## 5.	Feature Selection
 On the basis of correlation analysis and variance values, features were selected. The following features were removed: 
"pickup_datetime",
     "pickup_date",
    "dropoff_datetime",
     "passenger_count",
    'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
    'dropoff_latitude'
											
## 6.	Modelling
Applied Linear Regression algorithm first and got the following: 
Mean Squared Error: 0.0020959491839673324
R-squared (R2) Score: 0.6407131195910444

The results suggest that the linear regression model may not be the best choice for predicting trip duration in this dataset, as the model's performance is relatively poor.

Then applied LightGBM with hyperparameter tuning.
Employed the following parameters and applied GridSearchCV
param_grid = {
    'n_estimators': [100, 500],
    'learning_rate': [0.01,  0.2],
    'num_leaves': [31, 127],
    'max_depth': [5, 10],
    'min_child_samples': [10, 30]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')

## 7.	Preprocessing the Test set
## 8.	Saving the model file to use it in a  Flask WebApp. 

## Result
Achieved better model performance with the LightGBMRegressor:
•	Root Mean Squared Error (RMSE): 0.0414
•	R-squared (R2) Score: 0.7059
A lower RMSE and a higher R2 score indicate that the model is making more accurate predictions and explaining a significant portion of the variance in the data. These improved scores suggest that the LightGBM model is a good choice for regression task.
Hence this trained model can be used for making trip duration predictions. 

Made a Flask app and deployed this model. The app has a user friendly interface and doesn’t bothers the user with nitty gritty questions.
