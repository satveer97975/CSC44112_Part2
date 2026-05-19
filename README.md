# CSC44112_Part2
This project builds and compares multiple machine learning models to forecast weekly sales for 45 Walmart stores using historical data and contextual features such as holidays, temperature, fuel price, CPI, and unemployment rate.
The goal is to demonstrate a complete, end-to-end data science pipeline — from raw data exploration all the way to model evaluation and business interpretation.

Features used:
Store — Store number (1–45)
Date — Week start date
Weekly_Sales — Target variable (weekly revenue in USD)
Holiday_Flag — Whether the week contains a public holiday
Temperature — Regional average temperature (°F)
Fuel_Price — Regional fuel cost ($/gallon)
CPI — Consumer Price Index
Unemployment — Regional unemployment rate (%)

🚀 Future Improvements
 Time-series cross-validation (TimeSeriesSplit) for more realistic evaluation
 Per-store individual models for higher granularity
 LSTM or Facebook Prophet for true sequential forecasting
 Hyperparameter tuning with GridSearchCV or Bayesian optimisation
 REST API deployment with FastAPI + Docker
 SHAP values for explainability
