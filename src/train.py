import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import os

DATA_PATH = "data/dataset.csv"
MODEL_PATH = "models/model.pkl"

# Create models directory
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Species"])
y = df["Species"]

# Encode target labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y_encoded)

# Save model and encoder
joblib.dump(
    {"model": model, "encoder": encoder},
    MODEL_PATH
)

print("Iris model trained and saved successfully")
