import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import datetime


#  Page Configuration

st.set_page_config(
    page_title="Intelligent HVAC Energy Dashboard",
    page_icon="🔥",
    layout="wide"
)

st.title("🌡 Intelligent HVAC Energy Prediction & Control Dashboard")



#  Load Saved Models & Scaler


def load_joblib(path):
    return joblib.load(path) if Path(path).exists() else None


scaler = load_joblib("saved_models/scaler.pkl")
xgb_model = load_joblib("saved_models/xgb_model.pkl")
rf_model = load_joblib("saved_models/rf_model.pkl")
lgbm_model = load_joblib("saved_models/lgbm_model.pkl")
cat_model = load_joblib("saved_models/cat_model.pkl")


models = {
    "XGBoost": xgb_model,
    "Random Forest": rf_model,
    "LightGBM": lgbm_model,
    "CatBoost": cat_model,
}



#  Input Sidebar UI

st.sidebar.header("🔧 Input Parameters")

temp = st.sidebar.slider("Outdoor Temperature (°C)", 0, 50, 28)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 45)
sf = st.sidebar.number_input("Square Footage / Area", min_value=500, max_value=5000, value=1500)
occupancy = st.sidebar.slider("Occupancy (People)", 0, 300, 10)
renew = st.sidebar.slider("Renewable Energy Output (kW)", 0.0, 100.0, 10.0)
energy_last = st.sidebar.number_input("Last Hour Energy (kWh)", min_value=0.0, value=35.0)

# Time-based auto inputs
now = datetime.datetime.now()
day_of_year = now.timetuple().tm_yday
hour = now.hour

holiday = 1 if now.weekday() >= 5 else 0  # weekend



#  Prepare Data for Prediction

input_data = pd.DataFrame([{
    "Temperature": temp,
    "Humidity": humidity,
    "SquareFootage": sf,
    "Occupancy": occupancy,
    "HVACUsage": 1,             # default (encoded already)
    "LightingUsage": 0,
    "RenewableEnergy": renew,
    "DayOfWeek": now.weekday(),
    "Holiday": holiday,
    "Hour": hour,
    "DayOfYear": day_of_year,
    "Energy_Lag1": energy_last
}])

scaled_input = scaler.transform(input_data)



#  Prediction for All Models


results = {}

for name, model in models.items():
    if model:
        results[name] = float(model.predict(scaled_input)[0])

st.subheader("📊 Model Prediction Comparison (kWh)")
st.write(pd.DataFrame(results, index=["Energy Prediction (Next Hour)"]).T)



#  HVAC Decision Engine

best_model = min(results, key=results.get)
predicted_energy = results[best_model]

if predicted_energy > 35 :
    hvac_action = "⬆ AC ON – High Load Predicted"
    status_color = "red"
elif predicted_energy > 25:
    hvac_action = "⚠ Fan Only – Medium Load"
    status_color = "orange"
else:
    hvac_action = " HVAC OFF – Low Load & Optimized"
    status_color = "green"


st.header("🚦 HVAC Control Decision")
st.markdown(f"""
<div style="padding:20px;border-radius:10px;background-color:{status_color};color:white;">
<h2>{hvac_action}</h2>
<h3>Predicted Load: {predicted_energy:.2f} kWh</h3>
</div>
""", unsafe_allow_html=True)


#  Visualization

st.subheader("📈 Model Output Bar Chart")
st.bar_chart(pd.DataFrame(results, index=["Predicted kWh"]))

st.subheader("🔍 Input Preview (Model receives this)")
st.dataframe(input_data)

