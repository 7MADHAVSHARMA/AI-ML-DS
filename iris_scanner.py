# IRIS SCANNER - Flower Species Classifier

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df["species"] = df["target"].map({0:"Setosa",1:"Versicolor",2:"Virginica"})

# Features and labels
X = df.iloc[:, :-2]
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=10
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest Model
model = RandomForestClassifier(n_estimators=120, random_state=10)
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Evaluation
print("\nIRIS SCANNER RESULTS")
print("Accuracy:", round(accuracy_score(y_test, pred)*100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, pred, target_names=["Setosa","Versicolor","Virginica"]))

# Feature importance
plt.figure(figsize=(7,4))
plt.barh(iris.feature_names, model.feature_importances_)
plt.title("Feature Importance – Iris Scanner")
plt.xlabel("Importance Score")
plt.show()
