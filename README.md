\# Walmart Demand Forecasting App



A full-stack web application that forecasts consumer demand using Google Trends search-interest data, built as an end-to-end machine learning application — from raw data to a deployed, tested product.



\*\*Live demo:\*\* https://walmart-demand-forecast-app.onrender.com

\*(Free-tier hosting — the app may take 30–60 seconds to wake up if it hasn't been visited recently.)\*



\## Overview



This project uses Google Trends search-interest data for the keyword "Walmart" (United States, weekly frequency) as a proxy for real-world consumer demand, and forecasts the next 12 weeks using a trained machine learning model served through a REST API and visualized in an interactive browser dashboard.



\## What it does



\- Loads historical Google Trends data and engineers time-series features (lag values, rolling averages)

\- Trains a Random Forest Regression model to forecast future demand

\- Serves live 12-week forecasts through a Flask REST API (`/forecast`)

\- Renders the forecast as an interactive line chart in the browser using Chart.js

\- Includes automated tests (pytest) covering the API and feature engineering logic



\## Model development notes



The model was originally prototyped using an LSTM (TensorFlow). Given the relatively small size of the available time-series dataset, the LSTM tended to overfit/overshoot on unseen data. Based on this, the project was re-engineered around a Random Forest Regression model with lag-variable and rolling-average feature engineering, which was better suited to the dataset size and achieved a Mean Absolute Error (MAE) of ≈3.23 on the test set.



\## Tech stack



| Layer | Tools |

|---|---|

| Data \& Modeling | Python, Pandas, scikit-learn, TensorFlow (early prototyping) |

| Backend / API | Flask |

| Frontend | HTML, JavaScript, Chart.js |

| Testing | Pytest |

| Deployment | Render, Gunicorn |

| Version Control | Git, GitHub |



\## Running it locally



```bash

git clone https://github.com/chandu2020a-bit/walmart-demand-forecast-app.git

cd walmart-demand-forecast-app

python -m venv venv

venv\\Scripts\\activate   # on Windows

pip install -r requirements.txt

python demand\_forecast.py   # trains and saves the model

python app.py                # starts the local server

```



Then open `http://127.0.0.1:5000` in your browser.



\## Running tests



```bash

pytest

```



\## Possible future improvements



\- Recursive multi-step forecasting instead of reusing the last 12 rows' feature values, for improved accuracy on longer horizons

\- Add confidence intervals around the forecast

\- Support forecasting for additional keywords/regions beyond Walmart/US



\## Author



Chandu A

\[LinkedIn](https://linkedin.com/in/chandu-a-8925632ab) · \[GitHub](https://github.com/chandu2020a-bit)

