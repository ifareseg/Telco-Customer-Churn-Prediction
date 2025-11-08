import streamlit as st
import pandas as pd
import joblib

# ------------------------تحميل الموديل المدرب-------------------------
@st.cache_resource   #Streamlit بيحمّل الموديل مرة واحدة فقط، وبعدها يحتفظ به في الذاكرة.
def load_model():
    return joblib.load("best_churn_model.pkl")

model = load_model()


st.title("Telecom Customer Churn Prediction")
st.write("أدخل بيانات العميل لمعرفة إذا كان معرّضًا لترك الشركة (Churn).")

st.markdown("---")

st.subheader(" بيانات العميل")

# نفس الأعمدة اللي استخدمتها في X أثناء التدريب
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("SeniorCitizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, step=1)
phone_service = st.selectbox("PhoneService", ["Yes", "No"])
multiple_lines = st.selectbox("MultipleLines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("PaperlessBilling", ["Yes", "No"])
payment_method = st.selectbox("PaymentMethod",["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.number_input("MonthlyCharges", min_value=0.0, max_value=300.0, value=70.0, step=1.0)
total_charges = st.number_input("TotalCharges", min_value=0.0, max_value=10000.0, value=1000.0, step=10.0)

# ----------------------------------  تجميع المدخلات في DataFrame-------------------
input_dict = {
    "gender": [gender],
    "SeniorCitizen": [senior],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
}

input_df = pd.DataFrame(input_dict)

st.markdown("---")
st.subheader(" معاينة بيانات الإدخال")
st.dataframe(input_df)

st.markdown("---")

#  ----------------------------------التنبؤ----------------------------
st.subheader(" نتيجة التنبؤ")
#  تضيف سلايدر  threshold  :
threshold = st.slider("Threshold لاحتمال الـ Churn (افتراضي 0.5):", 0.1, 0.9, 0.5, 0.05)
if st.button("Predict"):     #---- الموديل Pipeline، فـ .predict و .predict_proba هيشتغلوا مباشرة على input_df
    proba = model.predict_proba(input_df)[0][1]  # احتمال Churn = 1
    pred = int(proba >= threshold)               # or model.predict(input_df)[0]

    st.write(f" Probability of Churn: **{proba:.2f}** ({proba*100:.1f} %)")

    if pred == 1:
        st.error(f"العميل **مُعرّض لترك الخدمة (Churn)**. (Threshold={threshold:.2f})")
    else:
        st.success(f"العميل **غير مُعرّض حاليًا لترك الخدمة**. (Threshold={threshold:.2f})")
