
import pandas as pd
import numpy as np


train_df = pd.read_csv("datas/raw/customer_churn_dataset-training-master.csv")
test_df = pd.read_csv("datas/raw/customer_churn_dataset-testing-master.csv")

#EDA 

print(train_df.head())
print(train_df.info())
print(train_df.describe())

print("\nMissing Values:\n", train_df.isnull().sum())
print("\nDuplicate Rows:", train_df.duplicated().sum())
print("\nDataset Shape:", train_df.shape)
print("\nColumn Names:", train_df.columns)

print(train_df.dtypes)
print(train_df["Churn"].value_counts())
print(train_df.describe(include="object"))

#VISUALIZATION 

import matplotlib.pyplot as plt
import seaborn as sns

train_df.boxplot(figsize=(15,6))
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(train_df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()

# SPLIT FEATURES & TARGET

X = train_df.drop("Churn", axis=1)
y = train_df["Churn"]

#TRAIN TEST SPLIT

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  HANDLE MISSING VALUES 

# Fix target (VERY IMPORTANT)

y_train = y_train.fillna(y_train.mode()[0])
y_test = y_test.fillna(y_train.mode()[0])

# Numerical columns

num_cols = X_train.select_dtypes(include=['int64','float64']).columns
for col in num_cols:
    mean_val = X_train[col].mean()
    X_train[col] = X_train[col].fillna(mean_val)
    X_test[col] = X_test[col].fillna(mean_val)

# Categorical columns

cat_cols = X_train.select_dtypes(include=['object']).columns
for col in cat_cols:
    X_train[col] = X_train[col].fillna("Unknown")
    X_test[col] = X_test[col].fillna("Unknown")

#  ENCODING

from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

#  SCALING 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Extra safety

X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

#  MODEL 

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)

# ------------------ PREDICTION ------------------
y_pred = model.predict(X_test)

# ------------------ EVALUATION ------------------
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

import numpy as np

print(np.isnan(X_train).sum())
print(np.isnan(X_test).sum())

from sklearn.metrics import roc_auc_score

y_prob = model.predict_proba(X_test)[:, 1]
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/logistic_regression.pkl")

print("Model saved successfully!")

