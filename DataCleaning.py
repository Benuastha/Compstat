import pandas as pd

# Load the dataset
file_path = 'nyc-property-sales.csv'
data = pd.read_csv(file_path)

# Preprocessing the dataset
# 1. Remove rows with null values
data_cleaned = data.dropna()

# 2. Drop unnecessary columns (e.g., 'APARTMENT NUMBER')
columns_to_drop = ['APARTMENT NUMBER']
data_cleaned = data_cleaned.drop(columns=columns_to_drop)

# 3. Encode categorical columns using one-hot encoding
categorical_columns = [
    'BOROUGH', 'NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 
    'TAX CLASS AT PRESENT', 'BUILDING CLASS AT PRESENT', 
    'TAX CLASS AT TIME OF SALE', 'BUILDING CLASS AT TIME OF SALE'
]
data_encoded = pd.get_dummies(data_cleaned, columns=categorical_columns, drop_first=True)

# Save the processed dataset
processed_file_path = 'nyc-property-sales-processed.csv'
data_encoded.to_csv(processed_file_path, index=False)

print(f"Processed dataset saved to {processed_file_path}")
