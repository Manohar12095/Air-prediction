import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Tamil Nadu AQI Predictor",
    page_icon="🌍",
    layout="centered"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
/* Full app background */
body {
    background: 
        linear-gradient(rgba(220, 245, 220, 0.9), rgba(220, 245, 220, 0.9)),
        url("https://www.transparenttextures.com/patterns/tree-bark.png");
    background-attachment: fixed;
}

/* Main white card */
.main {
    background-color: #f4fff4;
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.15);
}

/* Headings */
h1 {
    text-align: center;
    color: #1b5e20;
}

h2, h3 {
    color: #2e7d32;
}

/* Buttons */
.stButton > button {
    background-color: #66bb6a;
    color: white;
    border-radius: 10px;
    font-size: 16px;
}

.stButton > button:hover {
    background-color: #388e3c;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
rf_model = joblib.load("model_rf.pkl")

# ---------------- TITLE ----------------
st.title("🌫️ Tamil Nadu Air Quality Index Predictor")
st.write("Random Forest Machine Learning Model")

# ---------------- CITY PRESETS (TN DISTRICTS) ----------------
st.subheader("🏙️ Select District (Tamil Nadu)")

tn_districts = {
    "Custom": [10, 30, 20, 80],
    "Chennai": [12, 40, 30, 120],
    "Coimbatore": [8, 25, 18, 70],
    "Madurai": [10, 28, 22, 85],
    "Trichy": [9, 26, 20, 75],
    "Salem": [11, 30, 24, 90],
    "Erode": [7, 22, 16, 65],
    "Tiruppur": [9, 27, 21, 80],
    "Vellore": [10, 29, 23, 88],
    "Thanjavur": [6, 20, 15, 55],
    "Tirunelveli": [5, 18, 14, 50],
    "Thoothukudi": [7, 24, 18, 68],
    "Kanyakumari": [4, 15, 12, 45],
    "Cuddalore": [11, 32, 26, 95],
    "Villupuram": [9, 28, 22, 82],
    "Nagapattinam": [6, 21, 16, 58],
    "Dindigul": [8, 23, 18, 70],
    "Karur": [7, 22, 17, 65],
    "Namakkal": [8, 24, 19, 72],
    "Krishnagiri": [9, 26, 20, 78],
    "Dharmapuri": [8, 24, 19, 74],
    "Kanchipuram": [11, 33, 27, 98],
    "Tiruvallur": [12, 35, 28, 105],
    "Ranipet": [10, 30, 24, 90],
    "Chengalpattu": [11, 32, 26, 100],
    "Ariyalur": [6, 20, 15, 55],
    "Perambalur": [7, 22, 17, 60],
    "Pudukkottai": [7, 23, 18, 62],
    "Sivagangai": [6, 21, 16, 58],
    "Ramanathapuram": [5, 19, 14, 52],
    "Virudhunagar": [8, 25, 20, 75],
    "Theni": [6, 20, 15, 55],
    "Tenkasi": [5, 18, 14, 50],
    "Mayiladuthurai": [6, 21, 16, 58],
    "Kallakurichi": [8, 24, 19, 70]
}

district = st.selectbox("Choose District", list(tn_districts.keys()))
defaults = tn_districts[district]

# ---------------- INPUT SLIDERS ----------------
st.subheader("📊 AQI Component Inputs")

CO_AQI = st.slider("CO AQI Value", 0, 300, defaults[0])
OZONE_AQI = st.slider("Ozone AQI Value", 0, 300, defaults[1])
NO2_AQI = st.slider("NO₂ AQI Value", 0, 300, defaults[2])
PM25_AQI = st.slider("PM2.5 AQI Value", 0, 500, defaults[3])

input_data = np.array([[CO_AQI, OZONE_AQI, NO2_AQI, PM25_AQI]])

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict AQI"):
    with st.spinner("Analyzing air quality... 🌍"):
        prediction = rf_model.predict(input_data)[0]

    st.success(f"🌫️ Predicted AQI Value: **{prediction:.2f}**")
    st.balloons()

    # ---------------- AQI CATEGORY & HEALTH ADVISORY ----------------
    st.subheader("🩺 Health Advisory")

    if prediction <= 50:
        category = "Good 🟢"
        advice = "Air quality is good. Enjoy outdoor activities."
    elif prediction <= 100:
        category = "Moderate 🟡"
        advice = "Air quality is acceptable. Sensitive people should be cautious."
    elif prediction <= 200:
        category = "Poor 🟠"
        advice = "Reduce outdoor activities. Wear a mask."
    elif prediction <= 300:
        category = "Very Poor 🔴"
        advice = "Avoid outdoor activities. Health alert!"
    else:
        category = "Severe ☠️"
        advice = "Serious health risk. Stay indoors."

    st.write(f"**Category:** {category}")
    st.warning(advice)

    # ---------------- AQI SPEEDOMETER ----------------
    st.subheader("🎯 AQI Speedometer")

    fig, ax = plt.subplots()
    ax.barh(["AQI Level"], [prediction])
    ax.set_xlim(0, 500)
    ax.set_xlabel("AQI Scale")
    st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Tamil Nadu AQI Prediction System | Random Forest ML Project | Done by Manohar.S")
