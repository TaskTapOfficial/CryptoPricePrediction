import krakenex
import pandas as pd
import ta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import requests
import os
import datetime

# -----------------------------
# Sentiment Functions
# -----------------------------
def get_fear_greed():
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        resp = requests.get(url).json()
        return int(resp['data'][0]['value'])
    except:
        return 50

def get_news_sentiment():
    try:
        url = "https://cryptopanic.com/api/v1/posts/?auth_token=demo&currencies=XRP&kind=news"
        resp = requests.get(url).json()
        score, count = 0, 0
        for item in resp['results'][:10]:
            title = item['title'].lower()
            if any(w in title for w in ["lawsuit","ban","hack","drop","down","bearish"]):
                score -= 1
            elif any(w in title for w in ["win","approval","partnership","up","gain","rise","bullish"]):
                score += 1
            count += 1
        return score / count if count else 0
    except:
        return 0

# -----------------------------
# Kraken Data
# -----------------------------
k = krakenex.API()

def get_ohlc_data(pair="XRP/CAD", interval=1440, days=365):
    since = int((pd.Timestamp.today() - pd.Timedelta(days=days)).timestamp())
    resp = k.query_public('OHLC', {'pair': pair, 'interval': interval, 'since': since})
    ohlc = resp['result'][list(resp['result'].keys())[0]]
    df = pd.DataFrame(ohlc, columns=['time','open','high','low','close','vwap','volume','count'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df[['open','high','low','close','vwap','volume']] = df[['open','high','low','close','vwap','volume']].astype(float)
    return df

# -----------------------------
# Forecast Function
# -----------------------------
def run_forecast():
    df = get_ohlc_data("XRP/CAD", interval=1440, days=365)
    df = df[['time','close','volume']].set_index('time')

    # Technical Indicators
    df['EMA5'] = ta.trend.ema_indicator(df['close'], window=5)
    df['EMA15'] = ta.trend.ema_indicator(df['close'], window=15)
    df['EMA30'] = ta.trend.ema_indicator(df['close'], window=30)
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['MACD'] = ta.trend.macd(df['close'])
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['BB_high'] = bb.bollinger_hband()
    df['BB_low'] = bb.bollinger_lband()
    df['Lag1'] = df['close'].shift(1)
    df['Lag2'] = df['close'].shift(2)

    sentiment_score = get_news_sentiment()
    fear_greed = get_fear_greed()
    df['Sentiment'] = sentiment_score
    df['FearGreed'] = fear_greed
    df = df.dropna()

    # Train Models
    features = ['EMA5','EMA15','EMA30','RSI','MACD','BB_high','BB_low',
                'Lag1','Lag2','volume','Sentiment','FearGreed']
    X = df[features]
    y = df['close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, shuffle=False)

    rf_model = RandomForestRegressor(n_estimators=500, random_state=42)
    rf_model.fit(X_train, y_train)
    gb_model = GradientBoostingRegressor(n_estimators=300, random_state=42)
    gb_model.fit(X_train, y_train)

    latest_features = X.iloc[[-1]]
    rf_pred = rf_model.predict(latest_features)[0]
    gb_pred = gb_model.predict(latest_features)[0]
    final_pred = (rf_pred + gb_pred) / 2

    rf_y_pred = rf_model.predict(X_test)
    gb_y_pred = gb_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_y_pred)
    rf_r2 = r2_score(y_test, rf_y_pred)
    gb_mae = mean_absolute_error(y_test, gb_y_pred)
    gb_r2 = r2_score(y_test, gb_y_pred)

    # Sentiment Adjustment
    adjustment = 0
    if sentiment_score > 0.5 and fear_greed > 75:
        adjustment = 0.02
    elif sentiment_score < -0.5 and fear_greed < 30:
        adjustment = -0.02
    final_pred_adj = final_pred * (1 + adjustment)

    actual_today = df['close'].iloc[-1]

    # Daily Signal
    signal = "🟡 Neutral"
    reason = "Price expected to stay stable"
    if final_pred_adj > actual_today * 1.01 and sentiment_score >= 0 and fear_greed >= 55:
        signal = "🟢 Bullish"
        reason = "Forecast shows >1% gain with supportive sentiment"
    elif final_pred_adj < actual_today * 0.99 and (sentiment_score < 0 or fear_greed <= 45):
        signal = "🔴 Bearish"
        reason = "Forecast shows >1% drop with weak sentiment"

    # 7-Day Forecast
    future_preds = []
    temp_closes = df['close'].tolist()
    for i in range(7):
        temp_df = pd.DataFrame({'close': temp_closes[-30:]})
        temp_df['EMA5'] = ta.trend.ema_indicator(temp_df['close'], window=5)
        temp_df['EMA15'] = ta.trend.ema_indicator(temp_df['close'], window=15)
        temp_df['EMA30'] = ta.trend.ema_indicator(temp_df['close'], window=30)
        temp_df['RSI'] = ta.momentum.rsi(temp_df['close'], window=14)
        temp_df['MACD'] = ta.trend.macd(temp_df['close'])
        bb = ta.volatility.BollingerBands(temp_df['close'], window=20, window_dev=2)
        temp_df['BB_high'] = bb.bollinger_hband()
        temp_df['BB_low'] = bb.bollinger_lband()
        temp_df['Lag1'] = temp_df['close'].shift(1)
        temp_df['Lag2'] = temp_df['close'].shift(2)
        temp_df['volume'] = df['volume'][-30:].values
        temp_df['Sentiment'] = sentiment_score
        temp_df['FearGreed'] = fear_greed
        temp_df = temp_df.dropna()
        last_features = temp_df[features].iloc[[-1]]
        rf_next = rf_model.predict(last_features)[0]
        gb_next = gb_model.predict(last_features)[0]
        pred_price = ((rf_next + gb_next) / 2) * (1 + adjustment)
        future_preds.append(round(pred_price, 3))
        temp_closes.append(pred_price)

    weekly_signal = "🟡 Neutral"
    weekly_reason = "Prices expected to remain mostly stable this week"
    if future_preds[-1] > future_preds[0] * 1.02:
        weekly_signal = "🟢 Bullish"
        weekly_reason = "7-day forecast shows >2% growth"
    elif future_preds[-1] < future_preds[0] * 0.98:
        weekly_signal = "🔴 Bearish"
        weekly_reason = "7-day forecast shows >2% decline"

    results = {
        "Date": df.index[-1],
        "Actual_Close": round(actual_today, 3),
        "RF_Pred_Tomorrow": round(rf_pred, 3),
        "GB_Pred_Tomorrow": round(gb_pred, 3),
        "Final_Adj_Pred_Tomorrow": round(final_pred_adj, 3),
        "Trade_Signal": signal,
        "Signal_Reason": reason,
        "Weekly_Trend": weekly_signal,
        "Weekly_Reason": weekly_reason,
        "RF_MAE": round(rf_mae, 3),
        "RF_R2": round(rf_r2, 3),
        "GB_MAE": round(gb_mae, 3),
        "GB_R2": round(gb_r2, 3),
        "Sentiment": sentiment_score,
        "FearGreed": fear_greed,
        "Forecast_7Day": future_preds
    }

    # Only log locally
    # Log forecast (works locally and on Streamlit)
    # Log forecast (auto-create CSV with headers if missing)
    log_file = "xrp_cad_forecast_log.csv"
    log_entry = pd.DataFrame([results])

    try:
        if not os.path.exists(log_file):
            log_entry.to_csv(log_file, index=False)
        else:
            existing_log = pd.read_csv(log_file)
            updated_log = pd.concat([existing_log, log_entry], ignore_index=True)
            updated_log.to_csv(log_file, index=False)
    except Exception as e:
        print(f"⚠️ Could not write to log file: {e}")

    return results

if __name__ == "__main__":
    print(run_forecast())
