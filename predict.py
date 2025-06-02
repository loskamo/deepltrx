import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import asyncio
import websockets
import json
import numpy as np
from datetime import datetime

# ==== CONFIG ====
DATA_PATH = 'dataset.parquet'
MODEL_PATH = 'model_best.pt'
WINDOW_SIZE = 48
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SYMBOL = 'trxusdt'
STREAM_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL}@kline_1h"

# ==== MODEL ====
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.3, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1]).squeeze(-1)

# ==== LOAD MODEL + DATA ====
df = pd.read_parquet(DATA_PATH).dropna().reset_index(drop=True)
df = df.sort_values(by='timestamp')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour_of_day'] = df['timestamp'].dt.hour / 23.0
df['day_of_week'] = df['timestamp'].dt.dayofweek / 6.0

feature_cols = df.drop(columns=['timestamp', 'future_high', 'future_low']).columns
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

X_all = df[feature_cols].values.astype(np.float32)
X_tensor = torch.tensor(X_all[-WINDOW_SIZE:], dtype=torch.float32).unsqueeze(0).to(DEVICE)

model = LSTMModel(input_size=X_tensor.shape[-1]).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

with torch.no_grad():
    center_pred = model(X_tensor).item()
    lower = center_pred * 0.99
    upper = center_pred * 1.01

# ==== BINANCE WS LISTENER ====
async def listen_binance():
    async with websockets.connect(STREAM_URL) as ws:
        print(" Connected.")
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            k = data['k']
            o = float(k['o'])
            h = float(k['h'])
            l = float(k['l'])
            c = float(k['c'])

            output = f"\rOHLC: {o:.6f} / {h:.6f} / {l:.6f} / {c:.6f} | 🔮 Predicted range: [{lower:.6f} ... {upper:.6f}]"
            print(output, end='', flush=True)

asyncio.run(listen_binance())
