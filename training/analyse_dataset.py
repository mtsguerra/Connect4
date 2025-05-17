import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
data_path = os.path.join("data", "connect4_dataset.csv")
df = pd.read_csv(data_path)

# Basic statistics
print(f"Dataset shape: {df.shape}")
print(f"Number of unique moves: {df['move'].nunique()}")
print(f"Move distribution:\n{df['move'].value_counts(normalize=True)}")

# Visualize move distribution
plt.figure(figsize=(10, 6))
df['move'].value_counts().plot(kind='bar')
plt.title("Distribution of Moves in Dataset")
plt.xlabel("Column")
plt.ylabel("Count")
plt.savefig(os.path.join("data", "move_distribution.png"))
plt.close()

# Split features and target
X = df.drop('move', axis=1)
y = df['move']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
print("Training a Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Move")
plt.ylabel("True Move")
plt.savefig(os.path.join("data", "confusion_matrix.png"))
plt.close()

# Save the model
import joblib
model_path = os.path.join("data", "connect4_model.joblib")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")