# Lab Activity 1 - Data Preprocessing

## Description
This repository contains the implementation of data preprocessing for the **LoanData_Raw_v1.0.csv** dataset. The activity includes handling missing values and applying feature scaling to prepare the dataset for further analysis.

## Features
- **Data Cleaning:** Identifies and fills missing values using column means.
- **Feature Scaling:** Applies `StandardScaler` to normalize numerical features.
- **File Handling:** Reads the raw dataset and saves the cleaned, scaled data as a new CSV file.

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- Google Colab (if running on the cloud)

## Usage
1. **Mount Google Drive** (if using Colab) to access files.
2. **Load the dataset** from `LoanData_Raw_v1.0.csv`.
3. **Analyze missing values** and fill them with column means.
4. **Apply StandardScaler** for feature scaling.
5. **Save the preprocessed dataset** as `LoanData_Cleaned_Scaled.csv`.

## Running the Script
Execute the provided Python script in a Jupyter Notebook or Google Colab:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Mount Google Drive (if using Google Colab)
from google.colab import drive
drive.mount('/content/drive')

# Read the dataset
nfl_data = pd.read_csv("/content/LoanData_Raw_v1.0.csv")

# Set seed for reproducibility
np.random.seed(0)

# Display first few rows
print(nfl_data.head())

# Get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()
print(missing_values_count[0:10])

# Calculate percentage of missing data
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()
percent_missing = (total_missing / total_cells) * 100
print(f"Percentage of missing data: {percent_missing:.2f}%")

# Fill missing values with the column mean
nfl_data.fillna(nfl_data.mean(numeric_only=True), inplace=True)

# Apply StandardScaler to numerical columns
scaler = StandardScaler()
numeric_cols = nfl_data.select_dtypes(include=['number']).columns
nfl_data[numeric_cols] = scaler.fit_transform(nfl_data[numeric_cols])

# Save cleaned and scaled data
cleaned_file_path = "/content/LoanData_Cleaned_Scaled.csv"
nfl_data.to_csv(cleaned_file_path, index=False)
print(f"Cleaned and scaled data saved as '{cleaned_file_path}'.")
```

## Output
- **LoanData_Cleaned_Scaled.csv** – Preprocessed dataset ready for analysis.

## License
This project is for academic purposes only.
