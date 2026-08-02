import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

FEATURES = ["t", "lag_1", "lag_2", "lag_4", "rolling_4"]



# Load Google Trends demand data
df = pd.read_csv(
    "data/google_trends.csv",
    skiprows=3,
    names=["date", "demand"]
)

df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
df["t"] = range(len(df))

# Feature engineering
df["lag_1"] = df["demand"].shift(1)
df["lag_4"] = df["demand"].shift(4)
df["rolling_4"] = df["demand"].rolling(4).mean()
df["lag_2"] = df["demand"].shift(2)

df = df.dropna()

# Time-based train-test split
# -------------------- Time-based Train-Test Split --------------------

split_index = int(len(df) * 0.8)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

X_train = train[FEATURES]
y_train = train["demand"]

X_test = test[FEATURES]
y_test = test["demand"]



# Train ML model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Test prediction
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print("MAE:", mae)

import joblib
joblib.dump(model, "rf_model.pkl")
print("Model saved as rf_model.pkl")

# Forecast next 12 weeks
future = df.tail(12).copy()
future["t"] = range(df["t"].max() + 1, df["t"].max() + 13)

future_preds = model.predict(
    future[["t", "lag_1", "lag_2", "lag_4", "rolling_4"]]
)

# Plot forecast
# -------------------- Visualization --------------------

plt.figure(figsize=(12, 6))

# Historical data
plt.plot(
    df["date"],
    df["demand"],
    label="Historical Demand",
    linewidth=2
)

# Test predictions (last 12 weeks)
plt.plot(
    test["date"],
    preds,
    label="Model Predictions (Test Period)",
    linestyle="--",
    linewidth=2
)

# Future forecast
future_dates = pd.date_range(
    df["date"].iloc[-1],
    periods=13,
    freq="W"
)[1:]

plt.plot(
    future_dates,
    future_preds,
    label="Forecasted Demand",
    linestyle=":",
    linewidth=3
)

# Forecast start marker
plt.axvline(
    x=df["date"].iloc[-1],
    color="gray",
    linestyle="--",
    alpha=0.6,
    label="Forecast Start"
)

plt.title("AI-Based Market Demand Forecast (Walmart)", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Demand Index (Google Trends)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


