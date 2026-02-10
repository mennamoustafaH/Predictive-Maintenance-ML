# Predictive Maintenance (Machine Learning) + Streamlit App

This project builds machine learning models for predictive maintenance using sensor/process data.
It covers two tasks:

1) *Failure Prediction (Binary Classification)*  
   Predict whether the machine will fail (0 = No Failure, 1 = Failure)

2) *Failure Type Prediction (Multi-Class Classification)*  
   Predict the failure category *only when failure is expected*.

A lightweight *Streamlit app* is included so *non-programmers* can use the models through a simple form interface (no coding needed).

---

## Repository Files
- predictive_maintenance.csv → dataset
- machine predictive maintenance.ipynb → full ML workflow (training + evaluation)
- app.py → Streamlit app (user interface)
- requirements.txt → Python dependencies
- binary_model.pkl → trained binary model (failure / no failure)
- type_model.pkl → trained multi-class model (failure type)

---

## Dataset
File: predictive_maintenance.csv

Targets:
- *Binary target:* Target (0 = No Failure, 1 = Failure)
- *Multi-class target:* Failure Type (predicted only when Target = 1)

Features include sensor/process measurements such as temperatures, rotational speed, torque, and tool wear.

---

## Machine Learning Workflow
Implemented in: machine predictive maintenance.ipynb

Main steps:
1. Data loading & basic cleaning  
2. Feature selection + preprocessing (including encoding categorical variables when needed)  
3. Train/test split with stratification  
4. Handling class imbalance (e.g., SMOTE / class_weight)  
5. Model training:
   - Binary model uses predict_proba() to output failure probability
   - Failure-type model predicts the failure category (multi-class)
6. Evaluation:
   - precision / recall / F1-score
   - confusion matrix
   - model comparison and tuning
7. Model saving using joblib for reuse/deployment

---

## Run the Streamlit App Locally
### 1) Install dependencies
```bash
pip install -r requirements.txt
