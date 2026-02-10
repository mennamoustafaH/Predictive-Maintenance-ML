import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load models
# ----------------------------
binary_model = joblib.load("binary_model.pkl")   # 0/1 (Non-failure / Failure)
type_model   = joblib.load("failure_type_model.pkl")     # multiclass failure type

st.set_page_config(page_title="Predictive Maintenance App", layout="centered")
st.title("Predictive Maintenance – Failure & Failure Type Predictor")

st.write("Enter sensor values, then click *Predict*.")

# ----------------------------
features_num = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"]


st.subheader("Inputs")

inputs = {}

Machine_type=st.selectbox('Machine Type',['L','M','H'])
inputs['Type_L']=1 if Machine_type =='L' else 0
inputs['Type_M']=1 if Machine_type =='M'else 0

features_all=features_num+['Type_L','Type_M']

for col in features_num:
    inputs[col] = st.number_input(col, value=0.0)

# Build one-row DataFrame in the same order as training
X_input = pd.DataFrame([inputs], columns=features_all)

# ----------------------------
# Predict button
# ----------------------------
if st.button("Predict"):
    # 1) Binary prediction
    proba_fail = float(binary_model.predict_proba(X_input)[:, 1][0])
    pred_fail = int(proba_fail >= 0.25)  # default threshold 0.5 (you can change later)


    if pred_fail == 1:
        st.error("Prediction: *FAILURE*")
        st.subheader("Result (Binary)")
        st.write(f"*Failure Probability:* {proba_fail:.2%}")

        # 2) Failure type prediction (only if failure)
        type_pred = type_model.predict(X_input)[0]
        
        st.subheader("Result (Failure Type)")

        failure_map={0:'Heat Dissipation Failure',
                     1: 'Overstrain Failure',
                     2: 'Power Failure',
                     3: 'Random Failures',
                     4: 'Tool Wear Failure'}
        st.write('*Predicted Failure Type:*',failure_map.get(type_pred,'Not identified'))
        
      
    else:
        st.success("Prediction: *NO FAILURE*")
     
        
