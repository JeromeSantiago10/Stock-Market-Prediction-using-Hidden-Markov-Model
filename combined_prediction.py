import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import gradio as gr

# Global variables to store static graphs and predictions for both stocks
static_predictions = {
    "AAPL": {"dates": [], "prices": [], "plot": None},
    "IBM": {"dates": [], "prices": [], "plot": None}
}

# Function to load data
def load_data(stock_symbol):
    df = pd.read_csv(f'{stock_symbol}.csv')
    data = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Calculate features
    data['Return'] = data['Close'].pct_change()
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_30'] = data['Close'].rolling(window=30).mean()
    data['Volatility'] = data['Return'].rolling(window=10).std()
    data.dropna(inplace=True)

    lookback_period = 250
    data = data.tail(lookback_period)

    features = data[['Return', 'Volume', 'MA_10', 'MA_30', 'Volatility']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return data, features_scaled, scaler

# Function to train HMM
def train_hmm(features_scaled):
    model = hmm.GaussianHMM(n_components=4, covariance_type='diag', n_iter=1000)
    model.fit(features_scaled)
    return model

# Function to predict prices
def predict_prices(model, scaler, data, forecast_days):
    last_price = data['Close'].iloc[-1]
    predicted_prices = []
    dates = pd.date_range(start=data.index[-1] + pd.DateOffset(1), periods=forecast_days, freq='B')

    for _ in range(forecast_days):
        forecast_return = model.sample(1)[0]
        forecast_return_actual = scaler.inverse_transform(forecast_return)[0][0]
        predicted_next_price = last_price * (1 + forecast_return_actual)
        predicted_prices.append(predicted_next_price)
        last_price = predicted_next_price

    return dates, predicted_prices

# Function to generate the plot
def generate_static_plot(dates, predicted_prices, stock_symbol):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, predicted_prices, marker='o', color='orange')
    plt.title(f'{stock_symbol} Predicted Prices for Next 10 Days')
    plt.xlabel('Date')
    plt.ylabel('Predicted Price')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    plot_file = f"{stock_symbol}_static_plot.png"
    plt.savefig(plot_file)
    plt.close()

    return plot_file

# Function to prepare static graphs and data for both stocks
def prepare_static_data():
    for stock in ["AAPL", "IBM"]:
        data, features_scaled, scaler = load_data(stock)
        model = train_hmm(features_scaled)
        dates, predicted_prices = predict_prices(model, scaler, data, 10)
        static_predictions[stock]["dates"] = dates
        static_predictions[stock]["prices"] = predicted_prices
        static_predictions[stock]["plot"] = generate_static_plot(dates, predicted_prices, stock)

        # Generate dummy actual prices for evaluation
        actual_prices = np.random.normal(loc=predicted_prices[-1], scale=1, size=10)

        # Compute RMSE and MAE
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        mae = mean_absolute_error(actual_prices, predicted_prices)

        # Print RMSE and MAE
        print(f'{stock} RMSE: {rmse:.4f}')
        print(f'{stock} MAE: {mae:.4f}')

        # Calculate the Price Change Ratio (Close - Open) / Open
        predicted_df = pd.DataFrame({
            'Close': predicted_prices,
            'Open': [data['Close'].iloc[-1]] + predicted_prices[:-1]  # Open is the previous day's Close
        })
        predicted_df['Price_Change_Ratio'] = (predicted_df['Close'] - predicted_df['Open']) / predicted_df['Open']

        # Prior mean and variance for MAP
        mu_0 = 0
        sigma_mu = 0.01
        sample_mean = predicted_df['Price_Change_Ratio'].mean()
        sample_variance = predicted_df['Price_Change_Ratio'].var()
        n = len(predicted_df)

        sigma_likelihood = np.sqrt(sample_variance)
        mu_map = (n * sample_mean / sigma_likelihood*2 + mu_0 / sigma_mu2) / (n / sigma_likelihood2 + 1 / sigma_mu*2)

        # Print MAP estimate
        print(f'{stock} MAP estimate for (Close - Open) / Open: {mu_map:.4f}')

# Gradio function to fetch the static graph and predicted price
def stock_price_predictor(stock, day):
    # Fetch the static plot and prediction values
    plot_file = static_predictions[stock]["plot"]
    predicted_price_for_day = static_predictions[stock]["prices"][day - 1]
    return plot_file, predicted_price_for_day

# Prepare the static data and graphs once at the start
prepare_static_data()

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Stock Price Prediction")
    stock_selection = gr.Dropdown(choices=["AAPL", "IBM"], label="Select Stock")
    day_selection = gr.Slider(minimum=1, maximum=10, step=1, label="Select Day")
    predicted_price_display = gr.Number(label="Predicted Price for Selected Day")
    plot_output = gr.Image()

    # Function to update graph and price based on stock and day
    def update_interface(stock, day):
        plot, price = stock_price_predictor(stock, day)
        return plot, price

    stock_selection.change(fn=update_interface, inputs=[stock_selection, day_selection], outputs=[plot_output, predicted_price_display])
    day_selection.change(fn=update_interface, inputs=[stock_selection, day_selection], outputs=[plot_output, predicted_price_display])

# Launch the app
demo.launch()