# Final Project Proposal

**Date:** July 2026

---

## 1. Certificate Name

**Abdurahman Aden Mohamed**

---

## 2. Project Title and Description

**Title:** Customer Churn Prediction API

Subscription and service businesses lose money every time a paying customer walks away, and by the time someone notices, it's usually too late to do anything about it. This project builds a model that looks at a customer's account details, things like how long they've been signed up, what they pay, and what services they use, then flags who is likely to leave next. A customer support or retention team could use this to reach out to at-risk customers before they cancel, instead of finding out after the fact.

---

## 3. Problem Type

**Classification**: the target is binary, `Churn = Yes` or `Churn = No`.

This is supervised learning. The dataset already has historical records of who left and who stayed, so the model learns from those labeled examples.

---

## 4. Dataset

- **Source:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle).
- **Size:** The full dataset has 7,043 rows. I'll work with a random sample of about **2,000 rows** to keep training and iteration fast, which is still well above the 1,000-row minimum.
- **Target column:** `Churn`, whether the customer left in the last month (`Yes`/`No`).
- **Main features:**
  - `tenure`: how many months the customer has stayed
  - `MonthlyCharges` and `TotalCharges`: billing amounts
  - `Contract`: month-to-month, one year, or two year
  - `InternetService`: DSL, fiber optic, or none
  - `PaymentMethod`: how the customer pays
  - `OnlineSecurity`, `TechSupport`, `StreamingTV` and similar service add-ons: whether the customer subscribes to each
  - `gender`, `SeniorCitizen`, `Partner`, `Dependents`: basic demographic fields

**Preprocessing plan:** drop or impute missing values (a small number of blank `TotalCharges` rows are common in this dataset), encode categorical columns, scale numeric features, and split into an 80/20 train/test set, stratified on `Churn` so both sets keep a similar churn rate.

---

## 5. Algorithms I Plan to Train

| # | Algorithm | Why it fits |
| --- | --- | --- |
| 1 | **Logistic Regression** | Bootcamp baseline for binary classification. Coefficients are easy to explain to a non-technical stakeholder, like which factors push churn up or down. |
| 2 | **Random Forest** | Bootcamp ensemble method. Handles the mix of numeric and categorical features here without much tuning, and won't overfit as easily as a single tree. |
| 3 | **Gradient Boosting (scikit-learn)** | I'll research this one on my own. It tends to perform well on tabular data like this and is a reasonable candidate for the best-performing model. |

That covers the minimum of three. Two come from the bootcamp lessons, and Gradient Boosting is the one I'll dig into through the scikit-learn docs and a tutorial or two. I may add a Decision Tree or SVM as a fourth if time allows.

---

## 6. Evaluation Plan

**Metrics for all three models, measured on the same held-out test set:**

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

**Best-model rule:** I'll rank the models by **F1-Score**. Churn datasets are usually imbalanced, since far more people stay than leave, so accuracy alone can be misleading. F1 balances precision and recall, which matters here because missing an actual churner (low recall) and wrongly flagging a happy customer (low precision) are both costly in different ways. If two models land close on F1, I'll use Recall as the tiebreaker, since catching more real churners is the more useful outcome for a retention team.

---

## 7. Deployment Sketch

- **Framework:** FastAPI.
- **Endpoint:** `POST /predict`
- **Input JSON example:**

```json
{
  "tenure": 12,
  "MonthlyCharges": 70.35,
  "TotalCharges": 845.50,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "TechSupport": "No",
  "PaymentMethod": "Electronic check",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No"
}
```

- **Output JSON example:**

```json
{
  "prediction": "Churn",
  "probability": 0.78
}
```

The API loads the winning model from `models/best_model.pkl`, along with the fitted scaler and encoders, so incoming JSON gets preprocessed the same way the training data was.

---

## 8. Repository Plan

```
customer-churn-api/
├── dataset/
│   └── telco_churn_sample.csv
├── src/
│   ├── preprocess.py      # cleaning, encoding, scaling, train/test split
│   └── train.py           # trains all three models, prints comparison table, saves best
├── api/
│   └── app.py              # FastAPI app with /predict
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
├── README.md
├── requirements.txt
└── project_paper.md
```

**Planned run commands:**

```bash
python src/train.py
uvicorn api.app:app --reload
```

---
