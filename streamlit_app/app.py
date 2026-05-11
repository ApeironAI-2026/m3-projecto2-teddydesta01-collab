"""
Netflix Stock Price Forecasting — Streamlit App
Apeiron AI | Deep Learning Course
Run: streamlit run app.py
"""
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="NFLX Stock Forecaster", page_icon="📈", layout="centered")

# Model class (MUST match training)
class StockPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, model_type='lstm', dropout=0.0):
        super().__init__()
        self.model_type = model_type
        if model_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif model_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif model_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'best_stock_model.pth')
    config_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'config.json')

    device = torch.device('cpu')

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = StockPredictor(
            input_size=checkpoint.get('input_size', 1),
            hidden_size=checkpoint.get('hidden_size', 64),
            num_layers=checkpoint.get('num_layers', 2),
            model_type=checkpoint.get('model_type', 'lstm'),
            dropout=0.0
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        scaler_min = checkpoint.get('scaler_min', 0)
        scaler_max = checkpoint.get('scaler_max', 1)
        seq_length = checkpoint.get('seq_length', 60)
        model_type = checkpoint.get('model_type', 'lstm')
        st.sidebar.success(f"✅ {model_type.upper()} model loaded!")
    else:
        model = None
        scaler_min, scaler_max, seq_length, model_type = 0, 1, 60, 'lstm'
        st.sidebar.warning("⚠️ No trained model found.")

    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    return model, scaler_min, scaler_max, seq_length, model_type, config

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'NFLX.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
        return df
    return None

def predict_future(model, last_sequence, n_days, scaler_min, scaler_max):
    """Predict n_days into the future recursively."""
    model.eval()
    predictions = []
    current_seq = last_sequence.copy()

    with torch.no_grad():
        for _ in range(n_days):
            x = torch.FloatTensor(current_seq).unsqueeze(0).unsqueeze(-1)
            pred = model(x).item()
            predictions.append(pred)
            current_seq = np.append(current_seq[1:], pred)

    # Denormalize
    predictions = np.array(predictions) * (scaler_max - scaler_min) + scaler_min
    return predictions

# UI
st.title("📈 Netflix (NFLX) Stock Price Forecaster")
st.markdown("**Predict future stock prices using RNN/LSTM/GRU models trained on historical data.**")
st.markdown("*Powered by PyTorch | Apeiron AI*")
st.markdown("---")

model, scaler_min, scaler_max, seq_length, model_type, config = load_model()
df = load_data()

if df is not None:
    # Show historical data
    with st.expander("📊 Historical Data"):
        st.line_chart(df['Close'])
        st.dataframe(df.tail(10))

    # Model comparison (if available)
    if config and 'results' in config:
        with st.expander("🏆 Model Comparison Results"):
            results_df = pd.DataFrame(config['results']).T
            st.table(results_df.round(4))
            st.info(f"Best model: **{config.get('model_type', 'N/A').upper()}** (lowest RMSE)")

    # Prediction section
    st.markdown("### 🔮 Predict Future Prices")
    n_days = st.slider("Number of days to predict:", min_value=1, max_value=60, value=30)

    if st.button("🚀 Predict!") and model is not None:
        with st.spinner(f"Predicting {n_days} days with {model_type.upper()} model..."):
            # Get last seq_length days, normalize
            last_prices = df['Close'].values[-seq_length:]
            last_norm = (last_prices - scaler_min) / (scaler_max - scaler_min)

            future_prices = predict_future(model, last_norm, n_days, scaler_min, scaler_max)

            # Create date index for predictions
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days, freq='B')

            # Plot
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df.index[-120:], df['Close'].values[-120:], 'b-', label='Historical', linewidth=2)
            ax.plot(future_dates, future_prices, 'r--', label=f'Predicted ({n_days} days)', linewidth=2, marker='o', markersize=3)
            ax.axvline(x=last_date, color='gray', linestyle=':', alpha=0.7, label='Prediction Start')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.set_title(f'NFLX Price Forecast — {model_type.upper()} Model')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

            # Show predicted values
            pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price ($)': np.round(future_prices, 2)})
            st.dataframe(pred_df)

            col1, col2, col3 = st.columns(3)
            col1.metric("Last Known Price", f"${df['Close'].values[-1]:.2f}")
            col2.metric(f"Predicted (Day {n_days})", f"${future_prices[-1]:.2f}")
            change = ((future_prices[-1] - df['Close'].values[-1]) / df['Close'].values[-1]) * 100
            col3.metric("Change", f"{change:+.1f}%", delta=f"{change:+.1f}%")

    elif model is None:
        st.error("No trained model found. Train the model first using the notebook.")

else:
    st.error("NFLX.csv not found. Place it in the project root folder.")

# Disclaimer
st.markdown("---")
st.caption("⚠️ **Disclaimer:** This is for educational purposes only. Stock predictions are inherently uncertain. Do NOT use for actual trading decisions.")
st.markdown("*© 2026 Apeiron AI | 'Boundless Possibilities, Infinite Potential'*")
