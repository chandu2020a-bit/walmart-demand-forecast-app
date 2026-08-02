from flask import Flask, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("rf_model.pkl")

@app.route("/")
def home():
    return app.send_static_file("index.html")

FEATURES = ["t", "lag_1", "lag_2", "lag_4", "rolling_4"]

def load_and_prepare():
    df = pd.read_csv(
        "data/google_trends.csv",
        skiprows=3,
        names=["date", "demand"]
    )
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["t"] = range(len(df))
    df["lag_1"] = df["demand"].shift(1)
    df["lag_2"] = df["demand"].shift(2)
    df["lag_4"] = df["demand"].shift(4)
    df["rolling_4"] = df["demand"].rolling(4).mean()
    df = df.dropna()
    return df

@app.route("/forecast", methods=["GET"])
def forecast():
    df = load_and_prepare()

    future = df.tail(12).copy()
    future["t"] = range(df["t"].max() + 1, df["t"].max() + 13)

    future_preds = model.predict(future[FEATURES])

    future_dates = pd.date_range(
        df["date"].iloc[-1],
        periods=13,
        freq="W"
    )[1:]

    result = [
        {"date": str(d.date()), "forecast": float(p)}
        for d, p in zip(future_dates, future_preds)
    ]

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)