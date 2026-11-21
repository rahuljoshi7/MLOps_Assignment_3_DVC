import pandas as pd
import pickle
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load processed dataset
df = pd.read_csv("data/processed.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load model parameters
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)["model"]

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    random_state=params["random_state"]
)

model.fit(X_train, y_train)

# Save trained model
with open("models/model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model training completed successfully!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Model saved at: models/model.pkl")