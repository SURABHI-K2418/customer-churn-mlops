import pandas as pd
import os

print(os.getcwd())

train_df = pd.read_csv("data/raw/customer_churn_dataset-training-master.csv")
test_df = pd.read_csv("data/raw/customer_churn_dataset-testing-master.csv")

print(train_df.head())
print(train_df.info())
print(train_df.describe())


# Check missing values
print("\nMissing Values:")
print(train_df.isnull().sum())
# handle missing value
# Numeric columns
num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    train_df[col].fillna(train_df[col].mean(), inplace=True)

# Categorical columns
cat_cols = train_df.select_dtypes(include=['object']).columns

for col in cat_cols:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)

# Check again
print(train_df.isnull().sum())

# Check duplicate records
print("\nDuplicate Rows:")
print(train_df.duplicated().sum())

# Dataset shape check
print("\nDataset Shape:")
print(train_df.shape)

# Column names check
print("\nColumn Names:")
print(train_df.columns)

print(train_df.dtypes)
print(train_df["Churn"].value_counts())
print(train_df.describe(include="object"))
for col in train_df.columns:
    print(col, train_df[col].nunique())

import matplotlib.pyplot as plt

train_df.boxplot(figsize=(15,6))
plt.xticks(rotation=90)
plt.show()

import seaborn as sns

plt.figure(figsize=(12,8))
sns.heatmap(train_df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()

train_df.select_dtypes(include="object").columns       #categorical columns check
print(train_df.select_dtypes(include="object").columns)


X_train = train_df.drop("Churn", axis=1)
y_train = train_df["Churn"]

X_test = test_df.drop("Churn", axis=1)
y_test = test_df["Churn"]

# Remove rows where Churn is NaN
train_df = train_df.dropna(subset=["Churn"])

print("Train Churn NaN:", train_df["Churn"].isnull().sum())
print("Test Churn NaN:", test_df["Churn"].isnull().sum())

# Split
X = train_df.drop("Churn", axis=1)   
y = train_df["Churn"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(y_train.isnull().sum())
print(y_test.isnull().sum())

# Encoding 
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in X_train.select_dtypes(include=['object']).columns:
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

print("Encoding completed")

X_train.select_dtypes(include=['object', 'string']).columns

print(X_train.isnull().sum()[X_train.isnull().sum() > 0])

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# print(type(X_train))

       # \(X train and test NaN value check)\
# import numpy as np
# print(np.isnan(X_train).sum())
# print(np.isnan(X_test).sum())


# machine learning model training
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

#calculate metrics
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#calculate ROC-AUC score
from sklearn.metrics import roc_auc_score

y_prob = model.predict_proba(X_test)[:, 1]
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

#save the trained model
import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/logistic_regression.pkl")

print("Model saved successfully!")
















