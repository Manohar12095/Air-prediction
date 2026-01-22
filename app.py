import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Air Quality Index Predictor",
    page_icon="🌍",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1d2671, #c33764);
}
.main {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
}
h1, h2 {
    text-align: center;
    color: #1d2671;
}
.stButton>button {
    background-color: #1d2671;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
rf_model = joblib.load("model_rf.pkl")
nn_model = tf.keras.models.load_model("model_nn.h5")
scaler = joblib.load("scaler.pkl")

# ---------------- TITLE ----------------
st.title("🌍 Air Quality Index Predictor")
st.write("Predict AQI using Machine Learning & Neural Networks")

# ---------------- MODEL SELECTION ----------------
model_choice = st.radio(
    "Choose Prediction Model",
    ["Random Forest Model", "Neural Network Model"]
)

# ---------------- INPUT FIELDS ----------------
st.subheader("Enter Air Quality Parameters")

PM25 = st.number_input("PM2.5", 0.0)
PM10 = st.number_input("PM10", 0.0)
NO = st.number_input("NO", 0.0)
NO2 = st.number_input("NO2", 0.0)
NOx = st.number_input("NOx", 0.0)
NH3 = st.number_input("NH3", 0.0)
CO = st.number_input("CO", 0.0)
SO2 = st.number_input("SO2", 0.0)
O3 = st.number_input("O3", 0.0)
Benzene = st.number_input("Benzene", 0.0)
Toluene = st.number_input("Toluene", 0.0)
Xylene = st.number_input("Xylene", 0.0)

input_data = np.array([[PM25, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene]])

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict AQI"):
    if model_choice == "Random Forest Model":
        prediction = rf_model.predict(input_data)[0]

    else:
        input_scaled = scaler.transform(input_data)
        prediction = nn_model.predict(input_scaled)[0][0]

    st.success(f"🌫️ Predicted AQI Value: **{prediction:.2f}**")

    # AQI CATEGORY
    if prediction <= 50:
        st.info("🟢 Good")
    elif prediction <= 100:
        st.warning("🟡 Moderate")
    elif prediction <= 200:
        st.warning("🟠 Poor")
    elif prediction <= 300:
        st.error("🔴 Very Poor")
    else:
        st.error("☠️ Severe")
