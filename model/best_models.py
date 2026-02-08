import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

train = pd.read_csv("data/UNSW_NB15_training-set.csv")

TARGET = "label"
drop_cols = [TARGET, "attack_cat"]

X_train = train.drop(columns=drop_cols)
y_train = train[TARGET]

categorical_cols = ["proto", "service", "state"]
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols),
])

# Build full pipeline = preprocess + model
pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=120,
        max_depth=15,
        min_samples_leaf=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "model/best_random_forest.joblib")

print("Model saved as model/best_random_forest.joblib")
