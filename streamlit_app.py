import streamlit as st
import pandas as pd
import joblib

# ------------------------ Lad trained model -------------------------
@st.cache_resource  # Cache the model in memory for better performance
def load_model():
    return joblib.load("best_churn_model.pkl")

model = load_model()

# ------------------------ webpage Page  -------------------------
st.set_page_config(page_title="Project No.1: telecom customer churn prediction")
st.title("Project No.1: telecom Customer Churn Prediction")
st.write("Use this app to predict whether a telecom customer is likely to churn. y islam fares ahmed")
# ------------------------ Tabs: Single vs Batch -------------------------
tab_single, tab_batch = st.tabs(["Single customer prediction", "batch prediction from CSV file same"])

# ------------------------ Tab 1: Single Customer ------------------------
with tab_single:
    st.subheader("Single customer input")

    # Input fields (labels can be any text)
    gender = st.selectbox("gender", ["Male", "Female"])
    senior = st.selectbox("SeniorCitizen", [0, 1])
    partner = st.selectbox("partner", ["Yes", "No"])
    dependents = st.selectbox("dependents", ["Yes", "No"])
    tenure = st.number_input("tenure (months)", min_value=0, max_value=100, value=12, step=1)
    phone_service = st.selectbox("phoneservice", ["Yes", "No"])
    multiple_lines = st.selectbox("multipleLines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("internetservice", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("onlinesecurity", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("onlinebackup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("deviceprotection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("techSupport", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("streamingtv", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("streamingmovies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("paperlessbilling", ["Yes", "No"])
    payment_method = st.selectbox("paymentMethod",["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],)
    monthly_charges = st.number_input("monthlycharges", min_value=0.0, max_value=300.0, value=70.0)
    total_charges = st.number_input("totalcharges", min_value=0.0, max_value=10000.0, value=1000.0)

    # -------- vreate DataFrame from user input --------
    input_dict = {
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],          # fixed name
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
    st.subheader("Input preview")
    st.dataframe(input_df)
    st.subheader("Prediction result")
    if st.button("Predict (single customer)"):
        # model is a Pipeline (preprocessor + estimator)
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        st.write(f"**Churn probability:** {proba:.2f} ({proba*100:.1f}%)")
        if pred == 1:
            st.error("Customer is at risk of churn.")
        else:
            st.success("Customer is not at risk of churn.")

# ------------------------ Batch prediction from CSV ------------------------
with tab_batch:
    st.subheader("Batch prediction from CSV")
    st.write("Upload a CSV file that contains the same feature columns used during training exclude 'customerID' and 'Churn ")

    uploaded_file = st.file_uploader("Upload .CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df_batch.head())

            # Drop extra columns if present
            extra_cols = [c for c in ["customerID", "Churn"] if c in df_batch.columns]
            if extra_cols:
                df_batch = df_batch.drop(columns=extra_cols)
                st.info(f"Dropped extra columns: {extra_cols}")

            #    ترتيب الأعمدة بنفس ترتيب التدريب
            feature_cols = [
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "tenure",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
                "MonthlyCharges",
                "TotalCharges",
            ]

            missing = [c for c in feature_cols if c not in df_batch.columns]
            if missing:
                st.error(f"missing required columns in CSV: {missing}")
            else:
                df_batch = df_batch[feature_cols]

                # Predict
                predictions = model.predict(df_batch)
                probabilities = model.predict_proba(df_batch)[:, 1]

                results_df = df_batch.copy()
                results_df["Churn_Probability"] = probabilities
                results_df["Churn_Prediction"] = predictions

                st.write("Batch prediction results:")
                st.dataframe(results_df.head())

                # Download results
                csv_download = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download results as CSV",
                    data=csv_download,
                    file_name="churn_batch_predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error reading file: {e}")
