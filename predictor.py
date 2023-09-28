import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV

# Load the "Bengaluru_House_Data.csv" dataset
data = pd.read_csv('Bengaluru_House_Data.csv')

# Data Preprocessing
# Handling Missing Values
data.dropna(subset=['location', 'size', 'bath', 'balcony'], inplace=True)

# Handling 'total_sqft' column
def convert_sqft_to_numeric(sqft):
    if isinstance(sqft, str):
        parts = sqft.split('-')
        if len(parts) == 2:
            return (float(parts[0].strip()) + float(parts[1].strip())) / 2
        try:
            return float(sqft)
        except ValueError:
            return None
    return sqft

data['total_sqft'] = data['total_sqft'].apply(convert_sqft_to_numeric)

# Encoding Categorical Variables
label_encoder = LabelEncoder()
data['location_encoded'] = label_encoder.fit_transform(data['location'])

# Selecting Features
selected_features = ['location_encoded', 'total_sqft', 'bath', 'balcony']

X = data[selected_features]
y = data['price']

# Handling missing values in X
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Adding Polynomial Features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Model Training with Ridge Regression (L2 Regularization)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

pipeline.fit(X_train, y_train)

# Model Evaluation
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r_squared = r2_score(y_test, y_pred)

# Cross-Validation
cv_scores = cross_val_score(pipeline, X_poly, y, cv=5, scoring='neg_mean_squared_error')
cv_rmse = (-cv_scores)**0.5

# Print Evaluation Metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r_squared}")
print(f"Cross-Validation RMSE: {cv_rmse.mean()}")

# Hyperparameter Tuning using RandomizedSearchCV
param_dist = {'ridge__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=5, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
random_search.fit(X_poly, y)
print(f"Best Hyperparameters: {random_search.best_params_}")
