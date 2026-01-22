import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AQI Prediction – Random Forest",
    page_icon="🌍",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
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

# ---------------- LOAD MODEL ----------------
rf_model = joblib.load("model_rf.pkl")

# ---------------- TITLE ----------------
st.title("🌫️ Air Quality Index Prediction")
st.write("Using **Random Forest Machine Learning Model**")

# ---------------- INPUT SECTION ----------------
st.subheader("Enter Air Pollutant Values")

PM25 = st.number_input("PM2.5", min_value=0.0)
PM10 = st.number_input("PM10", min_value=0.0)
NO = st.number_input("NO", min_value=0.0)
NO2 = st.number_input("NO2", min_value=0.0)
NOx = st.number_input("NOx", min_value=0.0)
NH3 = st.number_input("NH3", min_value=0.0)
CO = st.number_input("CO", min_value=0.0)
SO2 = st.number_input("SO2", min_value=0.0)
O3 = st.number_input("O3", min_value=0.0)
Benzene = st.number_input("Benzene", min_value=0.0)
Toluene = st.number_input("Toluene", min_value=0.0)
Xylene = st.number_input("Xylene", min_value=0.0)

input_data = np.array([[PM25, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene]])

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict AQI"):
    if np.all(input_data == 0):
        st.error("⚠️ Please enter valid pollutant values (not all zeros)")
    else:
        prediction = rf_model.predict(input_data)[0]
        st.success(f"🌍 Predicted AQI Value: **{prediction:.2f}**")

        # AQI CATEGORY
        if prediction <= 50:
            st.info("🟢 Good – Air quality is satisfactory")
        elif prediction <= 100:
            st.warning("🟡 Moderate – Acceptable air quality")
        elif prediction <= 200:
            st.warning("🟠 Poor – Sensitive groups may be affected")
        elif prediction <= 300:
            st.error("🔴 Very Poor – Health alert")
        else:
            st.error("☠️ Severe – Serious health effects")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed as a Machine Learning Project | Random Forest Model")
