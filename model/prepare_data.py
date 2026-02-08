import pandas as pd

# 1) Load train and test files
train = pd.read_csv("../data/UNSW_NB15_training-set.csv")
test  = pd.read_csv("../data/UNSW_NB15_testing-set.csv")

# 2) Define target
TARGET = "label"

# 3) Features = everything except TARGET and attack_cat
drop_cols = [TARGET, "attack_cat"]
X_train = train.drop(columns=drop_cols)
y_train = train[TARGET]

X_test = test.drop(columns=drop_cols)
y_test = test[TARGET]

# 4) Print shapes to confirm
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)

# 5) Quick check: which columns are text/categorical?
print("\nCategorical columns:")
print(X_train.select_dtypes(include=["object"]).columns.tolist())