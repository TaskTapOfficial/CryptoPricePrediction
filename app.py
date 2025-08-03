import streamlit as st
import pandas as pd
import datetime
from xrp_cad_forecast import run_forecast

st.set_page_config(page_title="CryptoPricePrediction", layout="centered")

# ✅ SEO Meta Tags
st.markdown("""
<meta name="description" content="Free daily XRP price prediction and crypto forecast. Get tomorrow's XRP/CAD signals with AI-powered forecasts.">
<meta name="keywords" content="XRP prediction, crypto prediction, XRP price forecast, crypto forecast, XRP CAD tomorrow, cryptocurrency signals">
<meta name="robots" content="index, follow">
""", unsafe_allow_html=True)

# Title
st.title("📊 CryptoPricePrediction — Tomorrow's XRP/CAD Forecast")
st.write("Forecast updated daily — signals apply to **tomorrow's trading session**.")

if st.button("Run Forecast"):
    results = run_forecast()

    # Show forecast date info
    forecast_date = pd.to_datetime(results["Date"])
    tomorrow = forecast_date + pd.Timedelta(days=1)
    st.markdown(f"📅 **Forecast generated on {forecast_date.date()} for {tomorrow.date()}**")

    # Daily Signal
    st.subheader("🚦 Tomorrow's Trade Signal")
    st.metric(label="Signal", value=results["Trade_Signal"], delta=results["Signal_Reason"])

    # Weekly Trend
    st.subheader("📆 Weekly Trend")
    st.metric(label="Weekly Outlook", value=results["Weekly_Trend"], delta=results["Weekly_Reason"])

    # 7-Day Forecast with Real Dates
    st.subheader("📊 Full 7-Day Forecast")
    forecast_dates = [(tomorrow + pd.Timedelta(days=i)).date() for i in range(7)]
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Predicted Price (CAD)": results["Forecast_7Day"]
    })
    st.table(forecast_df)

# Affiliate Links
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
