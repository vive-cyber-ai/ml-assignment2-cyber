import pandas as pd
train=pd.read_csv("data/UNSW_NB15_training-set.csv")
print("rows and columns",train.shape)
print("\n columns in the dataset")
print(train.columns)
print("First 5 rows of the dataset")
print(train.head())
print("\n label distribution")
print(train["label"].value_counts())
