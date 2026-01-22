import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="AQI Prediction – Random Forest",
    page_icon="🌍",
    layout="centered"
)

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.main {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
}
h1 {
    color: #203a43;
    text-align: center;
}
.stButton>button {
    background-color: #2c5364;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Load model
rf_model = joblib.load("model_rf.pkl")

st.title("🌫️ AQI Prediction")
st.write("Using **Random Forest Machine Learning Model**")

st.subheader("Enter AQI Component Values")

# ✅ ONLY 4 INPUTS
CO_AQI = st.number_input("CO AQI Value", min_value=0.0)
OZONE_AQI = st.number_input("Ozone AQI Value", min_value=0.0)
NO2_AQI = st.number_input("NO2 AQI Value", min_value=0.0)
PM25_AQI = st.number_input("PM2.5 AQI Value", min_value=0.0)

input_data = np.array([[CO_AQI, OZONE_AQI, NO2_AQI, PM25_AQI]])

if st.button("🔍 Predict AQI"):
    if np.all(input_data == 0):
        st.error("⚠️ Please enter valid AQI values (not all zeros)")
    else:
        prediction = rf_model.predict(input_data)[0]
        st.success(f"🌍 Predicted AQI Value: **{prediction:.2f}**")

        if prediction <= 50:
            st.info("🟢 Good Air Quality")
        elif prediction <= 100:
            st.warning("🟡 Moderate Air Quality")
        elif prediction <= 200:
            st.warning("🟠 Poor Air Quality")
        elif prediction <= 300:
            st.error("🔴 Very Poor Air Quality")
        else:
            st.error("☠️ Severe Air Quality")
