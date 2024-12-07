import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load train and test features and target variables from CSV
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test = pd.read_csv('y_test.csv').squeeze()

# Define numerical and categorical features
numerical_features = ['LAND SQUARE FEET', 'GROSS SQUARE FEET', 'YEAR BUILT', 
                      'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS']
categorical_features = ['BOROUGH', 'NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 
                        'TAX CLASS AT PRESENT', 'TAX CLASS AT TIME OF SALE', 
                        'BUILDING CLASS AT PRESENT', 'BUILDING CLASS AT TIME OF SALE']

# Preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        # Add SimpleImputer for numerical features
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # Impute missing values
            ('scaler', StandardScaler())  # Scale numerical features
        ]), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # One-hot encode categorical features
    ]
)

# Combine preprocessing and model into a pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Preprocessing pipeline
    ('regressor', LinearRegression())  # Linear Regression model
])

# Fit the model
model_pipeline.fit(X_train, y_train)

print("Model training completed.")

# Evaluate the model
y_train_pred = model_pipeline.predict(X_train)
y_test_pred = model_pipeline.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Training R²: {train_r2:.3f}, Training MSE: {train_mse:.3f}")
print(f"Testing R²: {test_r2:.3f}, Testing MSE: {test_mse:.3f}")

# Get the preprocessing pipeline
preprocessor = model_pipeline.named_steps['preprocessor']
regressor = model_pipeline.named_steps['regressor']

# Get the feature names from the ColumnTransformer
numerical_features = preprocessor.transformers_[0][2]
categorical_encoder = preprocessor.transformers_[1][1]
categorical_feature_names = categorical_encoder.get_feature_names_out(categorical_features)

# Combine numerical and categorical feature names
feature_names = numerical_features + list(categorical_feature_names)

# Extract coefficients and map them to feature names
coefficients = regressor.coef_
coeff_summary = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# Display the coefficients sorted by their importance
coeff_summary = coeff_summary.sort_values(by='Coefficient', ascending=False)

# Save to Excel
coeff_summary.to_excel('linear_regression_coefficients.xlsx', index=False)