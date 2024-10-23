# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from fbprophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import yfinance as yf

# Step 1: Load the Dataset
# Replace 'AAPL' with the ticker symbol of your choice
ticker = 'AAPL'
data = yf.download(ticker, start='2015-01-01', end='2024-01-01')
data = data[['Close']]  # We only need the closing prices
data.reset_index(inplace=True)

# Step 2: Visualize the Data
plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Close'], label='Stock Price', color='blue')
plt.title(f'{ticker} Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()

# Step 3: ARIMA Model
# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data['Close'][:train_size], data['Close'][train_size:]

# Fit the ARIMA model
model = ARIMA(train, order=(5, 1, 0))  # You can change the order based on your analysis
arima_model = model.fit()
arima_forecast = arima_model.forecast(steps=len(test))

# Step 4: Facebook Prophet Model
# Prepare data for Prophet
prophet_data = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

# Split into training and testing sets
prophet_train = prophet_data[:train_size]
prophet_test = prophet_data[train_size:]

# Fit the Prophet model
prophet_model = Prophet()
prophet_model.fit(prophet_train)

# Make future predictions
future = prophet_model.make_future_dataframe(periods=len(prophet_test))
forecast = prophet_model.predict(future)

# Step 5: Evaluate Model Performance
# Calculate error metrics for ARIMA
y_true = test.values
y_pred_arima = arima_forecast.values

arima_mae = mean_absolute_error(y_true, y_pred_arima)
arima_rmse = np.sqrt(mean_squared_error(y_true, y_pred_arima))
arima_mape = mean_absolute_percentage_error(y_true, y_pred_arima)

print(f"ARIMA MAE: {arima_mae}")
print(f"ARIMA RMSE: {arima_rmse}")
print(f"ARIMA MAPE: {arima_mape}")

# Calculate error metrics for Prophet
y_pred_prophet = forecast['yhat'][train_size:].values

prophet_mae = mean_absolute_error(y_true, y_pred_prophet)
prophet_rmse = np.sqrt(mean_squared_error(y_true, y_pred_prophet))
prophet_mape = mean_absolute_percentage_error(y_true, y_pred_prophet)

print(f"Prophet MAE: {prophet_mae}")
print(f"Prophet RMSE: {prophet_rmse}")
print(f"Prophet MAPE: {prophet_mape}")

# Plot residuals for Prophet
prophet_residuals = y_true - y_pred_prophet

plt.figure(figsize=(10, 6))
plt.plot(prophet_residuals, label='Prophet Residuals', color='green')
plt.axhline(0, linestyle='--', color='black')
plt.title('Residuals for Prophet Model')
plt.legend()
plt.show()

# Plot residuals for ARIMA
arima_residuals = y_true - y_pred_arima

plt.figure(figsize=(10, 6))
plt.plot(arima_residuals, label='ARIMA Residuals', color='red')
plt.axhline(0, linestyle='--', color='black')
plt.title('Residuals for ARIMA Model')
plt.legend()
plt.show()
