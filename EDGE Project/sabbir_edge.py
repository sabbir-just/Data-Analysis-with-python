import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Load corrected dataset
path = "Social_Network_Ads (2).csv"
data = pd.read_csv(path)

# Ensure correct data types
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
data['EstimatedSalary'] = pd.to_numeric(data['EstimatedSalary'], errors='coerce')
data['Purchased'] = pd.to_numeric(data['Purchased'], errors='coerce')

# Handle missing data safely
data.loc[:, 'Age'] = data['Age'].fillna(data['Age'].mean())

# Check for class distribution
print("Class distribution before splitting:", np.bincount(data['Purchased']))

# Select features and target
x = data[['Age', 'EstimatedSalary']].values
y = data['Purchased'].values

# Split the dataset into training and testing sets (with stratification)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Ensure class distribution is maintained
print("Class distribution in y_train:", np.bincount(y_train))
print("Class distribution in y_test:", np.bincount(y_test))

# Apply MinMax scaling
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train SVM model with linear kernel
svm_model = SVC(kernel='linear', random_state=0)
svm_model.fit(x_train, y_train)

y_pred_svm = svm_model.predict(x_test)

# Evaluate SVM model
svm_cm = confusion_matrix(y_test, y_pred_svm, labels=[0, 1])
svm_acc = accuracy_score(y_test, y_pred_svm)
print("SVM Confusion Matrix:\n", svm_cm)
print(f"SVM Accuracy: {svm_acc:.2f}")

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=10, random_state=0)
rf_model.fit(x_train, y_train)

y_pred_rf = rf_model.predict(x_test)

# Evaluate Random Forest model
rf_cm = confusion_matrix(y_test, y_pred_rf, labels=[0, 1])
rf_acc = accuracy_score(y_test, y_pred_rf)
print("Random Forest Confusion Matrix:\n", rf_cm)
print(f"Random Forest Accuracy: {rf_acc:.2f}")

# Visualize SVM decision boundary (only works well for small feature spaces)
plt.figure(figsize=(10, 6))
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='winter', edgecolor='k', label='Training Data')
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred_svm, cmap='autumn', marker='x', label='Predicted Test Data')
plt.title('SVM Decision Boundary Visualization')
plt.legend()
plt.show()
