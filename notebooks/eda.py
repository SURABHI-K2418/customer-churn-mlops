import pandas as pd

train_df = pd.read_csv("data/raw/customer_churn_dataset-training-master.csv")
test_df = pd.read_csv("data/raw/customer_churn_dataset-testing-master.csv")

print(train_df.head())
print(train_df.info())
print(train_df.describe())