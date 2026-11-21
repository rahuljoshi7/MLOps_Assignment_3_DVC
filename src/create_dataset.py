import pandas as pd
from sklearn.datasets import load_breast_cancer
from pathlib import Path

# Load the Breast Cancer dataset
dataset = load_breast_cancer()

# Convert dataset into a DataFrame
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df["target"] = dataset.target

# Select the first 400 records for Dataset Version 1
df_v1 = df

# Save the dataset
output_path = Path("data/dataset.csv")
df_v1.to_csv(output_path, index=False)

print("Dataset V2 created successfully!")
print("Number of records:", len(df_v1))
print("Number of features:", len(dataset.feature_names))
print("Saved at:", output_path)