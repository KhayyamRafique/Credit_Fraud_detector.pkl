# Credit Card Fraud Detection

A machine learning model that classifies credit card transactions as 
**fraudulent** or **legitimate**, built as part of my learning journey 
into AI for offensive security.

## Overview
- **Dataset:** Credit Card Fraud Detection (Kaggle, by mlg-ulb)
- **Model:** Random Forest Classifier
- **Total transactions:** 284,807 (492 fraud, 284,315 legitimate)
- **Class imbalance:** 0.17% fraud — a real-world challenge in fraud detection

## Results
| Metric | Legitimate (0) | Fraud (1) |
|---|---|---|
| Precision | 1.00 | 0.77 |
| Recall | 1.00 | 0.97 |

**Overall accuracy: 99.99%** — but accuracy alone is misleading on 
imbalanced data. The key metric here is fraud recall (97%), meaning 
the model catches the vast majority of real fraud cases, at the cost 
of some false positives (77% precision).

## Key Learning
On highly imbalanced datasets, overall accuracy can hide poor 
minority-class performance. A trivial model that always predicts 
"legitimate" would already score ~99.83% accuracy while catching zero 
fraud. This project highlights why precision/recall per class matter 
more than accuracy alone in fraud/attack detection tasks.

## How to Use
```python
import joblib
model = joblib.load('credit_fraud_detector.pkl')

# Input must have the same 30 features as training data (Time, V1-V28, Amount)
prediction = model.predict(your_data)
```

## Tech Stack
Python, pandas, scikit-learn, joblib
