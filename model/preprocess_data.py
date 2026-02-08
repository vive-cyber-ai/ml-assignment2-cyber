import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# 1) Load data
train = pd.read_csv("../data/UNSW_NB15_training-set.csv")
test  = pd.read_csv("../data/UNSW_NB15_testing-set.csv")

TARGET = "label"
drop_cols = [TARGET, "attack_cat"]

X_train = train.drop(columns=drop_cols)
y_train = train[TARGET]

X_test = test.drop(columns=drop_cols)
y_test = test[TARGET]

# 2) Identify columns
categorical_cols = ['proto', 'service', 'state']

numeric_cols = [col for col in X_train.columns 
                if col not in categorical_cols]

print("Numeric columns count:", len(numeric_cols))
print("Categorical columns:", categorical_cols)

# 3) Create transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ]
)

# 4) Test the transformation
X_train_transformed = preprocessor.fit_transform(X_train)

print("\nShape after preprocessing:", X_train_transformed.shape)
