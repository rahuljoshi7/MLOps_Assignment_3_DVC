import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load processed dataset
df = pd.read_csv("data/processed.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Split dataset (same split used during training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load trained model
with open("models/model.pkl", "rb") as file:
    model = pickle.load(file)

# Make predictions
predictions = model.predict(X_test)

# Calculate evaluation metrics
metrics = {
    "accuracy": round(accuracy_score(y_test, predictions), 4),
    "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
    "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
    "f1_score": round(f1_score(y_test, predictions, zero_division=0), 4)
}

# Save metrics
with open("metrics/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Model evaluation completed successfully!")
print(json.dumps(metrics, indent=4))