import time
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier


# -------------------------------
# 1) Load data
# -------------------------------
TRAIN_PATH = "data/UNSW_NB15_training-set.csv"
TEST_PATH  = "data/UNSW_NB15_testing-set.csv"

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

TARGET = "label"
DROP_COLS = [TARGET, "attack_cat"]

X_train = train.drop(columns=DROP_COLS)
y_train = train[TARGET]

X_test = test.drop(columns=DROP_COLS)
y_test = test[TARGET]


# -------------------------------
# 2) Preprocessing (same for all models)
# -------------------------------
categorical_cols = ["proto", "service", "state"]
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols),
])


# -------------------------------
# 3) Helper: evaluate a trained model
# -------------------------------
def get_auc(model, X):
    """
    AUC needs continuous scores:
    - prefer predict_proba
    - else use decision_function
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def evaluate_model(name, pipeline):
    start = time.time()

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics based on predictions
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # AUC needs probabilities/scores
    scores = get_auc(pipeline, X_test)
    auc = roc_auc_score(y_test, scores) if scores is not None else float("nan")

    elapsed = time.time() - start

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc,
        "AUC": auc,
        "Time_sec": elapsed
    }


# -------------------------------
# 4) Define models (no hardcoding results)
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=15, random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=120,
        max_depth=15,
        min_samples_leaf=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    ),

    "kNN (slow)": KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    ),

    # GaussianNB needs dense arrays, so we convert inside pipeline
    "Naive Bayes": GaussianNB(),

    "XGBoost": XGBClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=2,
        reg_alpha=1,
        eval_metric="auc",
        n_jobs=-1,
        random_state=42
    )
}


# -------------------------------
# 5) Run all + collect results
# -------------------------------
results = []

for name, clf in models.items():
    # Special handling: GaussianNB needs dense matrix (toarray)
    if name == "Naive Bayes":
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("to_dense", FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
            ("clf", clf)
        ])
    else:
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("clf", clf)
        ])

    print(f"Training: {name} ...")
    results.append(evaluate_model(name, pipeline))

df = pd.DataFrame(results)

# Sort by AUC (best first)
df_sorted = df.sort_values(by="AUC", ascending=False)

print("\n===== MODEL COMPARISON (sorted by AUC) =====")
print(df_sorted.to_string(index=False))

# Optional: save to CSV for assignment report
df_sorted.to_csv("model/model_comparison.csv", index=False)
print("\nSaved: model/model_comparison.csv")
