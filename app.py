# file: app.py
import streamlit as st
import pandas as pd
import subprocess
import os

st.set_page_config(page_title="CryptoPricePrediction", layout="centered")

# Title
st.title("📊 CryptoPricePrediction — Daily XRP/CAD Forecast")
st.write("Machine-learning forecast with sentiment and weekly outlook.")

# Run your forecast script (when button clicked)
if st.button("Run Forecast"):
    result = subprocess.run(["python", "xrp_cad_forecast.py"], capture_output=True, text=True)
    st.text(result.stdout)

# Show latest log entry
if os.path.exists("xrp_cad_forecast_log.csv"):
    df = pd.read_csv("xrp_cad_forecast_log.csv")
    latest = df.tail(1)

    st.subheader("🚦 Daily Signal")
    st.metric(label="Trade Signal", value=latest["Trade_Signal"].values[0],
              delta=latest["Signal_Reason"].values[0])

    st.subheader("📆 Weekly Trend")
    st.metric(label="Weekly Outlook", value=latest["Weekly_Trend"].values[0],
              delta=latest["Weekly_Reason"].values[0])

    # Show Free 7-Day Forecast
    st.subheader("📊 7-Day Forecast")
    cols = ["Forecast_Day1","Forecast_Day2","Forecast_Day3",
            "Forecast_Day4","Forecast_Day5","Forecast_Day6","Forecast_Day7"]
    forecast = latest[cols].values[0]
    forecast_df = pd.DataFrame({
        "Day": [f"Day {i+1}" for i in range(7)],
        "Predicted Price (CAD)": forecast
    })
    st.table(forecast_df)

    # Tip: Later you can lock this section by wrapping in "if user_subscribed:" logic

# Affiliate Section
st.markdown("""
---
### 💰 Ready to Trade XRP/CAD?
👉 Start today with our trusted partners:

- [Trade on Kraken](https://your-kraken-affiliate-link.com)  
- [Get Free CAD Deposits with VirgoCX](https://your-virgocx-affiliate-link.com)  

*These are affiliate links — using them helps keep this forecast free.*
""")

# Disclaimer
st.markdown("""
---
⚠️ **Disclaimer**  
The forecasts on this site are for **educational and informational purposes only**.  
They do **not** constitute financial advice.  
Cryptocurrency trading is highly volatile and may result in loss of capital.  
Always consult a licensed financial advisor before making investment decisions.
""")
