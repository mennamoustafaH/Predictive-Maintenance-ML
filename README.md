# Predictive Maintenance (Machine Learning) + Streamlit Deployment
Machine learning models for predictive maintenance (binary failure + failure type classification)

This project builds machine learning models for predictive maintenance using sensor/process data.
It focuses on:
1) predicting whether a machine will fail (*binary classification*), and  
2) predicting the *failure type* (*multi-class classification*) only when failure is expected.

In addition, a lightweight *Streamlit app* is provided so *non-programmers* can use the trained models through a simple form-based interface (no coding needed).

---

## Dataset
- File: predictive_maintenance.csv
- Data contains machine/sensor measurements (e.g., temperatures, speed, torque, tool wear) and failure labels.

Targets:
- *Binary target:* Failure (0 = No Failure, 1 = Failure)
- *Multi-class target:* Failure Type (predicted only when Failure = 1)

---

## Machine Learning Workflow
Implemented in:
- machine predictive maintenance.ipynb

Main steps:
1. *Data loading & basic cleaning*
2. *Feature selection and preprocessing*
   - Ensuring model inputs match training feature columns
3. *Train/validation split*
4. *Class imbalance handling*
   - Using imbalanced-learn techniques to improve minority-class learning
5. *Model training*
   - *Binary model* outputs probability of failure (predict_proba)
   - *Failure-type model* predicts category when failure is predicted
6. *Evaluation*
   - Classification metrics (precision, recall, F1-score)
   - Confusion matrix and model comparison
7. *Model saving*
   - Models are exported using joblib for deployment/reuse

---

## Streamlit App (For Non-Programmers)
File:
- app.py

Purpose:
- Allows users to enter sensor values and get:
  - *Failure probability* (binary model)
  - *Predicted failure type* (only if failure is predicted)

Run locally:
```bash
streamlit run app.py
