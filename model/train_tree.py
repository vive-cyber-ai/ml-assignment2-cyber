import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score
)

train = pd.read_csv("data/UNSW_NB15_training-set.csv")
test  = pd.read_csv("data/UNSW_NB15_testing-set.csv")

TARGET = "label"
drop_cols = [TARGET, "attack_cat"]

X_train = train.drop(columns=drop_cols)
y_train = train[TARGET]

X_test = test.drop(columns=drop_cols)
y_test = test[TARGET]

categorical_cols = ["proto", "service", "state"]
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols),
])

X_train_t = preprocessor.fit_transform(X_train)
X_test_t  = preprocessor.transform(X_test)

model = DecisionTreeClassifier(max_depth=15, random_state=42)
model.fit(X_train_t, y_train)

y_pred = model.predict(X_test_t)
y_prob = model.predict_proba(X_test_t)[:,1]

print("\n===== DECISION TREE RESULTS =====")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("MCC      :", matthews_corrcoef(y_test, y_pred))
print("AUC      :", roc_auc_score(y_test, y_prob))
print("=================================\n")
