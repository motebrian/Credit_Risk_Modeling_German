# 1 Good (Lower Risk) and 0 Bad (Higher Risk)

import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_artifacts():
    model = joblib.load("extra_trees_credit_model.pkl")
    encoders = {col: joblib.load(f"{col}_encoder.pkl") for col in ('Sex', 'Housing', 'Saving accounts', 'Checking account')}
    return model, encoders

model, encoders = load_artifacts()

st.title("Credit Risk Prediction App")
st.write("Enter applicant information to predict the credit risk")

age = st.number_input("Age(19-75)",min_value=19, max_value = 75,value=30)
sex = st.selectbox('Sex', encoders['Sex'].classes_.tolist())
job = st.number_input('Job(0-3)',min_value=0,max_value=3, value=1)
housing = st.selectbox('Housing', encoders['Housing'].classes_.tolist())
savings_account = st.selectbox('Saving accounts', encoders['Saving accounts'].classes_.tolist())
checking_account = st.selectbox('Checking account', encoders['Checking account'].classes_.tolist())
credit_amount = st.number_input('Credit amount',min_value=0,value=100)
duration = st.number_input("Duration(months)",min_value=1,value=12)

if st.button("Predict Risk"):
    try:
        input_df = pd.DataFrame({
            'Age': [age],
            'Sex':[encoders['Sex'].transform([sex])[0]],
            'Job':[job],
            'Housing':[encoders['Housing'].transform([housing])[0]],
            'Saving accounts':[encoders['Saving accounts'].transform([savings_account])[0]],
            'Checking account':[encoders['Checking account'].transform([checking_account])[0]],
            'Credit amount': [credit_amount],
            "Duration":[duration]
        })
        pred = model.predict(input_df)[0]
        if pred == 1:
            st.success("The predicted credit risk is: **GOOD**")
        else:
            st.error("The predicted credit risk is: **BAD**")
    except Exception as e:
        st.error(f"Could not generate a prediction: {e}")