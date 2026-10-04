import pandas as pd

# Load dataset
data = pd.read_csv("processed.cleveland.data", header=None)

# Column names
data.columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

# Replace ? with missing value
data = data.replace("?", pd.NA)

# Convert columns to numeric
data = data.apply(pd.to_numeric, errors="coerce")

# Check missing values
print("Missing values before cleaning:")
print(data.isnull().sum())

# Remove rows containing missing values
data = data.dropna()

print("\nMissing values after cleaning:")
print(data.isnull().sum())

print("\nCleaned Dataset Shape:")
print(data.shape)

print("\nCleaned Dataset:")
print(data.head())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = data.drop("target", axis=1)
y = data["target"]

# Convert target into binary classification
# 0 = No Heart Disease
# 1 = Heart Disease
y = y.apply(lambda value: 1 if value > 0 else 0)

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Feature scaling completed successfully!")
from sklearn.linear_model import LogisticRegression

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")

# Make predictions
y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred)
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nModel Performance:")
print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)
print("Precision:", precision)
print("Recall:", recall)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disease", "Disease"]
)

display.plot()
plt.title("Heart Disease Prediction - Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get prediction probabilities
y_probability = model.predict_proba(X_test)[:, 1]

# Calculate ROC values
fpr, tpr, thresholds = roc_curve(y_test, y_probability)
auc_score = roc_auc_score(y_test, y_probability)

print("\nROC-AUC Score:", auc_score)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Heart Disease Prediction")
plt.legend()

plt.savefig("roc_curve.png")
plt.show()