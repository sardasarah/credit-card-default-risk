# Credit Card Default Risk Prediction: End-to-End Machine Learning Pipeline

This repository contains an institutional-grade Machine Learning pipeline designed to predict the probability of credit card default using the classic **UCI Default of Credit Card Clients Dataset**. 

The project transitions from exploratory data analysis to data engineering, robust pipeline training with XGBoost, and deploys a live interactive web interface using Streamlit.

## 🎯 Business Objectives & Financial Impact
From a risk-management perspective, the goal is to minimize financial losses by identifying high-risk clients before credit exploitation occurs.

### 💸 Business Impact Simulation (Based on Test Results):
Assuming an average financial loss of **$2,000** per undetected defaulter (False Negative):
* **Test Context:** Out of 6,000 evaluated clients, 1,327 were actual defaulters (representing a potential credit loss of **$2,654,000**).
* **Mitigated Loss:** The model successfully flagged 820 high-risk clients (**True Positives**), **mitigating $1,640,000 in direct bad debt**.
* **Managed Friction:** 929 good clients were flagged as risky (**False Positives**), representing a calculated risk trade-off to secure institutional capital.

---

## 🛠️ Data Engineering & Feature Engineering (Pandas)
To capture borrower behavior trends over the 6-month historical window, custom financial indicators were engineered using **Pandas**:
* **`LIMIT_UTILIZATION`**: Evaluates credit line dependency ($\text{Average Bill} / \text{Credit Limit}$). High utilization signals financial overextension.
* **`PAYMENT_TO_BILL_RATIO`**: Measures repayment health ($\text{Average Payment} / \text{Average Bill}$), distinguishing full-payers from dangerous revolvers.
* **Data Cleaning**: Consolidated undocumented categorical anomalies in `EDUCATION` and `MARRIAGE` into robust "Other" baselines to eliminate model noise.

---

## 🤖 Model Architecture & Pipeline (XGBoost)
To ensure production readiness and strictly prevent **Data Leakage**, a Scikit-Learn `Pipeline` was implemented:
1. **Stratified Split (80/20):** Maintained the 22% minority default class distribution across train and test sets.
2. **`ColumnTransformer`:** Automated `StandardScaler` for numeric scaling and `OneHotEncoder(drop='first')` for categorical tracking.
3. **Imbalance Handling:** Tuned `scale_pos_weight` inside the `XGBClassifier` to force the algorithm to prioritize high-sensitivity risk detection, achieving a strong **ROC-AUC Score of 0.7798**.

### 📊 Confusion Matrix Summary:
* **True Negatives (Approved Good Clients):** 3,744
* **True Positives (Blocked Bad Clients):** 820
* **False Positives (Rigorous Flags):** 929
* **False Negatives (Undetected Risk):** 507

---

## 🖥️ Streamlit Web Interface
The model is operationalized via a **Streamlit Web App**, allowing credit analysts or product managers to input customer financial data and receive an instantaneous risk assessment and default probability score.

### How to Run the Web Application:
1. Ensure dependencies are installed: `pip install -r requirements.txt`
2. Launch the server from your terminal:
   ```bash
   streamlit run app.py
