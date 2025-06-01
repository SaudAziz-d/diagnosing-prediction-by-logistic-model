import numpy as np
import pandas as pd
import seaborn as sns
import joblib

# Load the data
data = pd.read_csv("Diagnostic_Dataset.csv")

# Drop unnecessary columns
data.drop(['Unnamed: 32','Unnamed: 33','id'], axis=1, inplace=True)

# Convert diagnosis to binary (1: Malignant, 0: Benign)
data.diagnosis = [1 if value == 'M' else 0 for value in data.diagnosis]

# Convert diagnosis to categorical
data['diagnosis'] = data['diagnosis'].astype('category', copy=False)

# Plot class distribution
plot = data['diagnosis'].value_counts().plot(kind='bar', title='Class distributions \n (0: benign | 1: malignant)')
fig = plot.get_figure()

# Prepare data for training
y = data['diagnosis']
X = data.drop(['diagnosis'], axis=1)

# Normalize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Select and train the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Compare actual vs predicted
result = pd.DataFrame({'Actual': y_test, 'predicted': y_pred})
print(result.head())

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}\n")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'breast_cancer_model.pkl')
print("Model saved as 'breast_cancer_model.pkl'") 