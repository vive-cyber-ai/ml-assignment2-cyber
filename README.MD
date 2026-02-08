# 🛡️ Cyber Intrusion Detection – ML Assignment 2

## 1. Problem Statement

Network environments generate massive connection logs.  
The goal of this project is to build a **machine learning based Intrusion Detection System (IDS)** that can classify each network connection as:

- **Normal (0)**
- **Attack (1)**

The system is implemented end-to-end with:
- Data preprocessing  
- Training of 6 ML models  
- Comparison using standard metrics  
- Deployment through Streamlit web app

---

## 2. Dataset Description

**Dataset:** UNSW-NB15 Network Intrusion Dataset  

- Training samples: **82,332**  
- Testing samples: **175,341**  
- Total features: **45**

### Target Column
- `label` → 0 = Normal, 1 = Attack  
- `attack_cat` → attack type (not used for training)

### Feature Types
- Categorical: `proto`, `service`, `state`  
- Numerical: packet counts, bytes, TTL, load, jitter etc.

---

## 3. Preprocessing

1. Removed `label` and `attack_cat` from features  
2. One-Hot Encoding for categorical columns  
3. Standard Scaling for numerical columns  
4. Same pipeline reused for all models

---

## 4. Models Implemented

- Logistic Regression  
- Decision Tree  
- k-Nearest Neighbors  
- Naive Bayes  
- Random Forest  
- XGBoost

---

## 5. Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- MCC  
- AUC

---

## 6. Results Summary

| Model | AUC | Recall | Precision | MCC |
|------|-----|--------|-----------|-----|
| **Random Forest** | **0.916** | **0.69** | **0.97** | **0.60** |
| Logistic Regression | 0.807 | 0.25 | 0.85 | 0.19 |
| Naive Bayes | 0.784 | 0.44 | 0.99 | 0.44 |
| kNN | 0.689 | 0.40 | 0.93 | 0.35 |
| XGBoost | 0.765 | 0.15 | 0.77 | 0.07 |
| Decision Tree | 0.410 | 0.14 | 0.49 | -0.19 |

### Observation

- **Random Forest performed best** due to:
  - ability to handle high-dimensional one-hot features  
  - robustness to noise  
  - ensemble voting reducing overfitting  

- Logistic Regression was too linear  
- kNN was slow for large data  
- Single Decision Tree overfit  
- XGBoost required heavy tuning  
- Naive Bayes gave high precision but moderate recall

---

## 7. Streamlit Application

### Features
- Upload CSV  
- Threshold slider to control risk  
- Attack / Normal prediction  
- Risk score  
- Confusion matrix (if labels present)  
- Download results

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
