import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Advanced AQI Prediction System",
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
h1, h2 {
    text-align: center;
    color: #243b55;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
rf_model = joblib.load("model_rf.pkl")
nn_model = tf.keras.models.load_model("model_nn.h5")
scaler = joblib.load("scaler.pkl")

# ---------------- TITLE ----------------
st.title("🌫️ Advanced Air Quality Index Predictor")

# ---------------- CITY PRESETS ----------------
st.subheader("🏙️ City Presets")

city_data = {
    "Custom": [10, 30, 20, 80],
    "Chennai": [12, 35, 25, 90],
    "Delhi": [40, 60, 50, 180],
    "Mumbai": [18, 40, 30, 100],
    "Bangalore": [8, 25, 15, 60]
}

city = st.selectbox("Select a City", list(city_data.keys()))
default_values = city_data[city]

# ---------------- INPUT SLIDERS ----------------
st.subheader("📊 Enter AQI Component Values")

CO_AQI = st.slider("CO AQI Value", 0, 300, default_values[0])
OZONE_AQI = st.slider("Ozone AQI Value", 0, 300, default_values[1])
NO2_AQI = st.slider("NO₂ AQI Value", 0, 300, default_values[2])
PM25_AQI = st.slider("PM2.5 AQI Value", 0, 500, default_values[3])

input_rf = np.array([[CO_AQI, OZONE_AQI, NO2_AQI, PM25_AQI]])

# ---------------- MODEL SELECTION ----------------
model_choice = st.radio(
    "Choose Model",
    ["Random Forest", "Neural Network"]
)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict AQI"):
    if model_choice == "Random Forest":
        prediction = rf_model.predict(input_rf)[0]
    else:
        input_nn = scaler.transform(input_rf)
        prediction = nn_model.predict(input_nn)[0][0]

    st.success(f"🌍 Predicted AQI Value: **{prediction:.2f}**")

    # ---------------- AQI CATEGORY + HEALTH ADVISORY ----------------
    st.subheader("🩺 Health Advisory")

    if prediction <= 50:
        category = "Good 🟢"
        advice = "Air quality is good. Enjoy outdoor activities."
    elif prediction <= 100:
        category = "Moderate 🟡"
        advice = "Acceptable air quality. Sensitive people should be cautious."
    elif prediction <= 200:
        category = "Poor 🟠"
        advice = "Reduce outdoor activities. Wear a mask if needed."
    elif prediction <= 300:
        category = "Very Poor 🔴"
        advice = "Avoid outdoor activities. Health alert!"
    else:
        category = "Severe ☠️"
        advice = "Serious health risk. Stay indoors."

    st.write(f"**Category:** {category}")
    st.warning(advice)

    # ---------------- AQI SPEEDOMETER (BAR STYLE) ----------------
    st.subheader("🎯 AQI Speedometer")

    gauge_df = pd.DataFrame({
        "AQI": [prediction]
    })

    fig, ax = plt.subplots()
    ax.barh(["AQI Level"], gauge_df["AQI"])
    ax.set_xlim(0, 500)
    ax.set_xlabel("AQI Scale")
    st.pyplot(fig)

    # ---------------- FEATURE IMPORTANCE ----------------
    st.subheader("📈 Feature Importance (Random Forest)")

    feature_names = ["CO AQI", "Ozone AQI", "NO2 AQI", "PM2.5 AQI"]
    importances = rf_model.feature_importances_

    fig2, ax2 = plt.subplots()
    ax2.bar(feature_names, importances)
    ax2.set_ylabel("Importance Score")
    ax2.set_title("Pollutant Impact on AQI")
    st.pyplot(fig2)

    # ---------------- MODEL COMPARISON ----------------
    st.subheader("⚖️ Model Comparison")

    rf_pred = rf_model.predict(input_rf)[0]
    nn_pred = nn_model.predict(scaler.transform(input_rf))[0][0]

    compare_df = pd.DataFrame({
        "Model": ["Random Forest", "Neural Network"],
        "Predicted AQI": [rf_pred, nn_pred]
    })

    fig3, ax3 = plt.subplots()
    ax3.bar(compare_df["Model"], compare_df["Predicted AQI"])
    ax3.set_ylabel("AQI Value")
    ax3.set_title("Model Prediction Comparison")
    st.pyplot(fig3)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Advanced AQI Prediction System | ML & Deep Learning Project")
