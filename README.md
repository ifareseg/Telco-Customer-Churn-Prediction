#  Telecom Customer Churn Prediction App

###  Overview
This project aims to predict whether a telecom customer is likely to **churn** (leave the company) based on their demographic, contract, and billing information.  
It uses **machine learning models** and a **Streamlit web app** for interactive predictions.

---

##  Features
- **End-to-end ML pipeline** with preprocessing, encoding, and scaling.
- Trained and compared **8 classification models**:
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
  - Gradient Boosting  
  - LightGBM  
  - CatBoost  
  - Support Vector Machine (SVM)  
  - K-Nearest Neighbors (KNN)
- **Automatic model selection** based on the best Recall/F1-score.
- **Interactive Streamlit interface** for:
  - Single customer input prediction  
  - Batch prediction via CSV file upload
- Visual comparison of model performance (Recall & F1-score Barplots).
- Confusion matrix visualization for the best model.

---

## Tech Stack
- **Language:** Python  
- **Libraries:** Streamlit, Scikit-learn, XGBoost, LightGBM, CatBoost, Pandas, NumPy, Seaborn, Matplotlib  
- **Model Deployment:** Streamlit Cloud / Hugging Face Spaces  
- **File Storage:** Joblib (for trained model persistence)

---

##  How to Run Locally
Clone the repository and install dependencies:

```bash
git clone https://github.com/DrIslam/telecom-churn-streamlit.git
cd telecom-churn-streamlit
pip install -r requirements.txt
streamlit run app.py
