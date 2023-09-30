# Bengaluru House Price Prediction

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24%2B-orange)
![Pandas](https://img.shields.io/badge/Pandas-1.0%2B-yellow)
![NumPy](https://img.shields.io/badge/NumPy-1.18%2B-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.1%2B-blue)
![Seaborn](https://img.shields.io/badge/Seaborn-0.10%2B-orange)

Predicting house prices in Bengaluru using machine learning.

## Overview

This project focuses on predicting house prices in Bengaluru, India, based on various features such as location, size, number of bathrooms, and balconies. It uses machine learning techniques to build a regression model capable of providing accurate price estimates for residential properties.

## Data

The dataset used for this project is sourced from 'Bengaluru_House_Data.csv.' It contains information about properties in Bengaluru, including their attributes and corresponding prices.

## Data Preprocessing

- Handling Missing Values: Rows with missing values in the 'location,' 'size,' 'bath,' and 'balcony' columns are removed.
- Handling 'total_sqft' column: The 'total_sqft' column is converted to numeric, handling different formats.
- Encoding Categorical Variables: Location data is label-encoded for model compatibility.
- Feature Selection: Relevant features including 'location_encoded,' 'total_sqft,' 'bath,' and 'balcony' are selected.

## Model Training

- Data Splitting: The dataset is split into training and testing sets for model evaluation.
- Polynomial Features: Polynomial features of degree 2 are added to capture complex relationships.
- Ridge Regression: A Ridge Regression model with L2 regularization is trained to predict house prices.
- Pipeline: A Scikit-Learn pipeline is used to streamline data preprocessing and model training.

## Model Evaluation

The trained model is evaluated using the following metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R2)

## Cross-Validation

Cross-validation is performed to assess model performance across multiple folds. The root mean squared error (RMSE) is used as the evaluation metric.

## Hyperparameter Tuning

RandomizedSearchCV is employed to fine-tune the hyperparameters of the Ridge Regression model, optimizing its predictive accuracy.

## Usage

1. Clone this repository.
2. Ensure you have the required libraries installed (`pip install -r requirements.txt`).
3. Run the Python script to predict house prices.

## Feedback and Contact

For any questions, feedback, or clarifications, please feel free to reach out via [GitHub](https://github.com/The-Ark-Knight) or [LinkedIn](https://www.linkedin.com/in/abhishekkumar-03).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
