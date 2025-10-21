import pandas as pd

# Path to your dataset
file_path = "Dataset/tcg_sea_dataset_2024_2025.csv"

# Load it
df = pd.read_csv(file_path)

# Basic info
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nColumn names:")
print(df.columns.tolist())

# Quick preview
print("\nSample data:")
print(df.head(5))
