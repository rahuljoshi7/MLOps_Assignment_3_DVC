import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("data/dataset.csv")

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create processed dataset
processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df["target"] = y.values

# Save processed dataset
processed_df.to_csv("data/processed.csv", index=False)

print("Preprocessing completed successfully!")
print("Processed records:", len(processed_df))
print("Processed features:", len(X.columns))