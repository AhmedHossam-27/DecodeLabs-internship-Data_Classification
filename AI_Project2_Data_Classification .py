
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("💡 Starting AI Pipeline for Project 2...")

iris_data = load_iris()

df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
df['target'] = iris_data.target

print("\n📊 First 5 rows of the dataset:")
print(df.head())

X = iris_data.data    # الميزات الأربعة: Sepal/Petal Length & Width
y = iris_data.target  # نوع الزهرة (Setosa, Versicolor, Virginica)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n✂️ Data Split Completed:")
print(f"   - Training set size: {X_train.shape[0]} samples")
print(f"   - Testing set size: {X_test.shape[0]} samples")


print("\n🤖 Training the KNN model...")

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==================================================")
print("🏆 MODEL PERFORMANCE RESULT")
print("==================================================")
print(f"🎯 KNN Model Accuracy : {accuracy * 100:.2f}%")
print("==================================================")


print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📝 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris_data.target_names))