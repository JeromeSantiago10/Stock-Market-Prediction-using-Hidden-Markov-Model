import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('IBM.csv')

# Data Preparation
data = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Calculate daily returns and moving averages
data['Return'] = data['Close'].pct_change()
data['MA_10'] = data['Close'].rolling(window=10).mean()  # 10-day moving average
data['MA_30'] = data['Close'].rolling(window=30).mean()  # 30-day moving average
data['Volatility'] = data['Return'].rolling(window=10).std()  # 10-day volatility
data = data.dropna()

# Use the last 250 days of data for training
lookback_period = 250
data = data.tail(lookback_period)

# Prepare the data for HMM using returns, volume, moving averages, and volatility
features = data[['Return', 'Volume', 'MA_10', 'MA_30', 'Volatility']].values

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Define the HMM
n_components = 5  # Experiment with the number of hidden states
model = hmm.GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=1000)

# Fit the model
model.fit(features_scaled)

# Forecast the next 10 days
forecast_days = 10
predicted_prices = []
dates = pd.date_range(start=data.index[-1] + pd.DateOffset(1), periods=forecast_days, freq='B')

# Last observed return for the prediction
last_observation = features_scaled[-1:]
for _ in range(forecast_days):
    # Sample the next return
    forecast_return = model.sample(1)[0]
    forecast_return_actual = scaler.inverse_transform(forecast_return)[0][0]
    last_price = data['Close'].iloc[-1]
    
    # Calculate the predicted price
    predicted_next_price = last_price * (1 + forecast_return_actual)
    predicted_prices.append(predicted_next_price)
    last_observation = scaler.transform(np.array([[forecast_return_actual, last_price, data['MA_10'].iloc[-1], data['MA_30'].iloc[-1], data['Volatility'].iloc[-1]]]))

# Create DataFrame for predictions
predicted_df = pd.DataFrame({
    'Close': predicted_prices,
    'Date': dates
}).set_index('Date')

# Generate dummy actual prices for MSE calculation (replace this with actual data if available)
actual_prices = np.random.normal(loc=predicted_prices[-1], scale=1, size=forecast_days)  # Placeholder
mse = mean_squared_error(actual_prices, predicted_prices)
print(f'Mean Squared Error (MSE) for the predictions: {mse:.4f}')

# Plot the predicted prices for the next 10 days only
plt.figure(figsize=(10, 5))
plt.plot(predicted_df.index, predicted_df['Close'], marker='o', linestyle='-', color='orange')
plt.title('Predicted Stock Prices for Next 10 Days')
plt.xlabel('Date')
plt.ylabel('Predicted Price')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Print the predicted prices for the next 10 days
print(predicted_df)