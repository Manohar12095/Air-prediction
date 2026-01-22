import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AQI Predictor – Random Forest",
    page_icon="🌍",
    layout="centered"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #141e30, #243b55);
}
.main {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
}
h1 {
    color: #243b55;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
rf_model = joblib.load("model_rf.pkl")

# ---------------- TITLE ----------------
st.title("🌫️ Air Quality Index Prediction")
st.write("Random Forest Machine Learning Model")

st.markdown("### 👉 Adjust the sliders to enter air quality values")

# ---------------- INPUTS (SLIDERS) ----------------
col1, col2 = st.columns(2)

with col1:
    CO_AQI = st.slider("CO AQI Value", 0, 300, 10)
    OZONE_AQI = st.slider("Ozone AQI Value", 0, 300, 30)

with col2:
    NO2_AQI = st.slider("NO₂ AQI Value", 0, 300, 20)
    PM25_AQI = st.slider("PM2.5 AQI Value", 0, 500, 80)

input_data = np.array([[CO_AQI, OZONE_AQI, NO2_AQI, PM25_AQI]])

# ---------------- PREDICT BUTTON ----------------
if st.button("🔍 Predict AQI"):
    if np.all(input_data == 0):
        st.error("⚠️ Please move the sliders to enter values")
    else:
        prediction = rf_model.predict(input_data)[0]

        st.success(f"🌍 Predicted AQI Value: **{prediction:.2f}**")

        # AQI CATEGORY
        if prediction <= 50:
            category = "Good 🟢"
        elif prediction <= 100:
            category = "Moderate 🟡"
        elif prediction <= 200:
            category = "Poor 🟠"
        elif prediction <= 300:
            category = "Very Poor 🔴"
        else:
            category = "Severe ☠️"

        st.subheader(f"AQI Category: {category}")

        # ---------------- GRAPH ----------------
        st.markdown("### 📊 Air Quality Component Analysis")

        df = pd.DataFrame({
            "Pollutant": ["CO AQI", "Ozone AQI", "NO2 AQI", "PM2.5 AQI"],
            "Value": [CO_AQI, OZONE_AQI, NO2_AQI, PM25_AQI]
        })

        fig, ax = plt.subplots()
        ax.bar(df["Pollutant"], df["Value"])
        ax.set_ylabel("AQI Value")
        ax.set_title("User Input AQI Components")

        st.pyplot(fig)
