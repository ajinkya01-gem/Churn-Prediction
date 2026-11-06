import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Telco Customer Churn Predictor", page_icon="📊", layout="wide")
st.title("📊 Telco Customer Churn Prediction App")
st.markdown("Enter customer details below to predict churn probability in real time.")

@st.cache_resource
def load_artifacts():
    model = joblib.load("best_churn_model.joblib")
    pipeline = joblib.load("preprocessing_pipeline.joblib")
    return model, pipeline

try:
    model, pipeline = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found! Please run `python main.py` or `churn_analysis.ipynb` first.")
    st.stop()

def predict_customer_churn(raw_input: dict):
    df_in = pd.DataFrame([raw_input])
    df_in['tenure_group'] = pd.cut(
        df_in['tenure'], bins=[-1, 12, 48, 100], labels=['0-12 months', '13-48 months', '>48 months']
    ).astype(str)
    df_in['monthly_charge_group'] = pd.cut(
        df_in['MonthlyCharges'], bins=[0, 35, 70, 150], labels=['Low', 'Medium', 'High']
    ).astype(str)
    df_in['has_dependents'] = (df_in['Dependents'] == 'Yes').astype(int)
    df_in['total_services'] = (df_in[pipeline['service_cols']] == 'Yes').sum(axis=1)

    df_encoded = pd.get_dummies(df_in, columns=pipeline['categorical_cols'], drop_first=True, dtype=int)
    for col in pipeline['feature_names']:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[pipeline['feature_names']]

    num_cols = pipeline['numerical_cols']
    df_encoded[num_cols] = pipeline['scaler'].transform(df_encoded[num_cols])

    prob = model.predict_proba(df_encoded)[0][1]
    pred = int(prob >= 0.5)
    return pred, prob

st.sidebar.header("Customer Demographics & Account")
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
tenure = st.sidebar.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Subscribed Services")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with col2:
    st.subheader("Billing & Contract")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=75.0, step=0.5)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(tenure * monthly_charges), step=10.0)

st.markdown("---")
if st.button("🔍 Predict Churn Risk", type="primary", use_container_width=True):
    user_input = {
        'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner,
        'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone_service,
        'MultipleLines': multiple_lines, 'InternetService': internet_service,
        'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
        'DeviceProtection': device_protection, 'TechSupport': tech_support,
        'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
        'Contract': contract, 'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    prediction, probability = predict_customer_churn(user_input)
    if prediction == 1:
        st.error(f"⚠️ **High Churn Risk Detected!** (Churn Probability: **{probability:.1%}**)")
        st.info("💡 **Recommended Action:** Offer a 1-year contract discount or complimentary Tech Support / Online Security bundle.")
    else:
        st.success(f"✅ **Low Churn Risk — Customer Likely to Stay** (Churn Probability: **{probability:.1%}**)")
