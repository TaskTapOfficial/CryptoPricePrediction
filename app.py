import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from xrp_cad_forecast import run_forecast

st.set_page_config(page_title="CryptoPricePrediction", layout="centered")

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

    # 7-Day Forecast
    st.subheader("📊 Full 7-Day Forecast")
    forecast_df = pd.DataFrame({
        "Day": [f"Day {i+1}" for i in range(7)],
        "Predicted Price (CAD)": results["Forecast_7Day"]
    })
    st.table(forecast_df)

    # Accuracy Chart (Last 7 Days)
    st.subheader("📈 Forecast Accuracy — Last 7 Days")
    try:
        df_log = pd.read_csv("xrp_cad_forecast_log.csv")
        df_log["Date"] = pd.to_datetime(df_log["Date"], errors="coerce")
        df_recent = df_log.tail(7)

        if not df_recent.empty:
            plt.figure(figsize=(8, 4))
            plt.plot(df_recent["Date"], df_recent["Actual_Close"], marker='o', label="Actual Price")
            plt.plot(df_recent["Date"], df_recent["Final_Adj_Pred_Tomorrow"], marker='x', label="Predicted Price")
            plt.title("XRP/CAD Forecast vs Actual — Last 7 Days")
            plt.xlabel("Date")
            plt.ylabel("Price (CAD)")
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
        else:
            st.info("⚠️ Not enough data yet to show accuracy chart. Run the forecast for several days first.")

    except Exception as e:
        st.info(f"⚠️ Accuracy chart unavailable: {e}")

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
