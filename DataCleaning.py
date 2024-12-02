# Load the dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
file_path = "nyc-property-sales.csv" 
data = pd.read_csv(file_path)

# Columns to convert
numeric_columns = ['LAND SQUARE FEET', 'GROSS SQUARE FEET', 'YEAR BUILT', 'SALE PRICE']

# Convert to numeric
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna(subset=['SALE PRICE'])

# Impute missing values with the median for numerical columns
impute_columns = ['LAND SQUARE FEET', 'GROSS SQUARE FEET', 'YEAR BUILT']
for col in impute_columns:
    data[col] = data[col].fillna(data[col].median())

print(data[numeric_columns].isna().sum())


#Handle Categorical Data
# Ensure categorical columns are strings
categorical_columns = ['BOROUGH', 'NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 'TAX CLASS AT PRESENT']
for col in categorical_columns:
    data[col] = data[col].astype(str)

#Dropping APARTMENT NUMBER

# Drop irrelevant columns
irrelevant_columns = ['APARTMENT NUMBER']
data = data.drop(columns=irrelevant_columns, errors='ignore')

# Columns to exclude
columns_to_exclude = ['SALE PRICE', 'EASE-MENT', 'ADDRESS', 'SALE DATE']

# Select all other columns as features
X = data.drop(columns=columns_to_exclude, errors='ignore')
y = data['SALE PRICE']

#Handle Numerical and Categorical Columns
numerical_features = ['LAND SQUARE FEET', 'GROSS SQUARE FEET', 'YEAR BUILT', 
                      'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS']
categorical_features = ['BOROUGH', 'NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 
                        'TAX CLASS AT PRESENT', 'TAX CLASS AT TIME OF SALE', 
                        'BUILDING CLASS AT PRESENT', 'BUILDING CLASS AT TIME OF SALE']
# Preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  # Scale numerical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # One-hot encode categorical features
    ]
)

#Train-Test Split
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of the split data
print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Save train and test features and target variables to CSV
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Train and test data saved to CSV files.")

