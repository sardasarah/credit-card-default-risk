import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Credit Risk Assessment", page_icon="💳", layout="centered")

st.title("💳 Credit Card Default Risk Predictor")
st.markdown("This application uses a trained XGBoost Machine Learning pipeline to evaluate the probability of a client defaulting on their next credit card payment.")

@st.cache_resource
def load_model():
    return joblib.load('credit_model_pipeline.pkl')

model = load_model()

st.header("👤 Client Financial Profile")

col1, col2, col3 = st.columns(3)

with col1:
    limit_bal = st.number_input("Credit Limit ($)", min_value=1000, max_value=1000000, value=50000, step=5000)
    age = st.number_input("Age (Years)", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")

with col2:
    education = st.selectbox("Education Level", options=[1, 2, 3, 4], format_func=lambda x: {1:"Graduate School", 2:"University", 3:"High School", 4:"Others"}[x])
    marriage = st.selectbox("Marital Status", options=[1, 2, 3], format_func=lambda x: {1:"Married", 2:"Single", 3:"Others"}[x])
    pay_0 = st.slider("Payment Status (Current Month)", min_value=-1, max_value=8, value=0, help="-1: Paid on time, 1+: Months of delay")

with col3:
    pay_2 = st.slider("Payment Status (1 Month Ago)", min_value=-1, max_value=8, value=0)
    pay_3 = st.slider("Payment Status (2 Months Ago)", min_value=-1, max_value=8, value=0)

st.subheader("📊 Recent Financial History")
avg_bill = st.number_input("Average Bill Amount Over Last 6 Months ($)", min_value=0, value=15000)
avg_pay = st.number_input("Average Payment Amount Over Last 6 Months ($)", min_value=0, value=15000)

if st.button("Evaluate Credit Risk", type="primary"):
    
    limit_utilization = avg_bill / (limit_bal + 1)
    payment_to_bill_ratio = avg_pay / avg_bill if avg_bill > 0 else 1.0
    
    # Criar o DataFrame com o formato EXACTO que o pipeline espera (1 elemento por coluna)
    input_data = pd.DataFrame({
        'LIMIT_BAL': [limit_bal], 
        'SEX': [sex], 
        'EDUCATION': [education], 
        'MARRIAGE': [marriage], 
        'AGE': [age],
        'PAY_0': [pay_0], 
        'PAY_2': [pay_2], 
        'PAY_3': [pay_3], 
        'PAY_4': [pay_3], # Replicando o histórico para as colunas remanescentes
        'PAY_5': [pay_3], 
        'PAY_6': [pay_3],
        'BILL_AMT1': [avg_bill], # Corrigido: sem o *6
        'BILL_AMT2': [avg_bill], # Corrigido: sem o *6
        'BILL_AMT3': [avg_bill], # Corrigido: sem o *6
        'BILL_AMT4': [avg_bill], # Corrigido: sem o *6
        'BILL_AMT5': [avg_bill], # Corrigido: sem o *6
        'BILL_AMT6': [avg_bill], # Corrigido: sem o *6
        'PAY_AMT1': [avg_pay],   # Corrigido: sem o *6
        'PAY_AMT2': [avg_pay],   # Corrigido: sem o *6
        'PAY_AMT3': [avg_pay],   # Corrigido: sem o *6
        'PAY_AMT4': [avg_pay],   # Corrigido: sem o *6
        'PAY_AMT5': [avg_pay],   # Corrigido: sem o *6
        'PAY_AMT6': [avg_pay],   # Corrigido: sem o *6
        'AVG_BILL': [avg_bill], 
        'LIMIT_UTILIZATION': [limit_utilization], 
        'PAYMENT_TO_BILL_RATIO': [payment_to_bill_ratio]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0, 1]
    
    st.markdown("---")
    st.subheader("🎯 Risk Assessment Result")
    
    if prediction == 1:
        st.error(f"⚠️ **HIGH RISK DETECTED (Reject Credit)**")
        st.metric(label="Probability of Default", value=f"{probability * 100:.2f}%")
    else:
        st.success(f"✅ **LOW RISK (Approve Credit)**")
        st.metric(label="Probability of Default", value=f"{probability * 100:.2f}%")