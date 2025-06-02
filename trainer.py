from datetime import datetime, timezone
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler

# ==== CONFIG ====
DATA_PATH = 'dataset.parquet'
MODEL_PATH = 'model_best.pt'
LOG_PATH = 'training_log.json'
WINDOW_SIZE = 48
BATCH_SIZE = 128
EPOCHS = 20
LR = 0.001
HIDDEN_SIZE = 256
NUM_LAYERS = 6
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==== DATASET ====
class TimeSeriesDataset(Dataset):
    def __init__(self, df, window_size, scaler):
        df['hour_of_day'] = df['timestamp'].dt.hour / 23.0
        df['day_of_week'] = df['timestamp'].dt.dayofweek / 6.0

        self.scaler = scaler
        self.features = df.drop(columns=['timestamp', 'future_high', 'future_low']).values.astype(np.float32)
        fh = df['future_high'].values.astype(np.float32)
        fl = df['future_low'].values.astype(np.float32)
        self.targets_center = (fh + fl) / 2
        self.targets_width = (fh - fl) / 2
        self.window_size = window_size

    def __len__(self):
        return len(self.features) - self.window_size

    def __getitem__(self, idx):
        X = self.features[idx:idx + self.window_size]
        center = self.targets_center[idx + self.window_size]
        width = self.targets_width[idx + self.window_size]
        y = np.array([center, width], dtype=np.float32)
        return torch.tensor(X), torch.tensor(y)

# ==== MODEL ====
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, dropout=0.3, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1]).squeeze(-1)  # (batch,)

# ==== LOSS + METRIC ====
def custom_loss(center_pred, center_true, half_width_true, lambda_cov=0.1):
    mse = nn.functional.mse_loss(center_pred, center_true)
    allowed_half_width = center_pred * 0.01
    mask = (half_width_true > allowed_half_width).float()
    penalty = mask.mean()
    return mse + lambda_cov * penalty

def compute_success_rate(center_pred, target):
    true_center = target[:, 0]
    true_half_width = target[:, 1]
    allowed_half_width = center_pred * 0.01
    success_mask = (true_half_width <= allowed_half_width)
    return success_mask.float().mean().item()

# ==== TRAIN ====
def evaluate_model(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            preds.append(pred)
            targets.append(y)
    center_preds = torch.cat(preds)
    targets = torch.cat(targets)
    success_rate = compute_success_rate(center_preds, targets)
    true_centers = targets[:, 0]
    true_half_width = targets[:, 1]
    allowed_half_width = center_preds * 0.01
    abs_errors = torch.abs(center_preds - true_centers)

    print(f"   [VAL] -  Success Rate: {success_rate*100:.2f}%")
    print(f"      ↪ Avg center_pred:     {center_preds.mean().item():.6f}")
    print(f"      ↪ Avg center_true:     {true_centers.mean().item():.6f}")
    print(f"      ↪ Avg abs error:       {abs_errors.mean().item():.6f}")
    print(f"      ↪ Avg allowed ±1%:     {allowed_half_width.mean().item():.6f}")
    print(f"      ↪ Avg actual half-wid: {true_half_width.mean().item():.6f}")

    return success_rate

def train(model, train_loader, val_loader, optimizer):
    best_score = -1.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            center_true = y[:, 0]
            half_width_true = y[:, 1]
            center_pred = model(X)
            loss = custom_loss(center_pred, center_true, half_width_true)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{EPOCHS}")
        val_score = evaluate_model(model, val_loader)
        if val_score > best_score:
            best_score = val_score
            best_state = model.state_dict()

    return best_score, best_state

# ==== UTILS ====
def load_data():
    df = pd.read_parquet(DATA_PATH).dropna().reset_index(drop=True)
    df = df.sort_values(by='timestamp')
    
    # Додаємо нові фічі
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['timestamp'].dt.hour / 23.0
    df['day_of_week'] = df['timestamp'].dt.weekday / 6.0

    scaler = StandardScaler()
    features = df.drop(columns=['timestamp', 'future_high', 'future_low'])
    df[features.columns] = scaler.fit_transform(features)
    return df, scaler


def save_log(score):
    log = {
        'score': score,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'model_path': MODEL_PATH
    }
    with open(LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2)

# ==== MAIN ====
def main(resume=True):
    print(" Loading data...")
    df, scaler = load_data()
    dataset = TimeSeriesDataset(df, WINDOW_SIZE, scaler)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    input_size = df.drop(columns=['timestamp', 'future_high', 'future_low']).shape[1]
    model = LSTMModel(input_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    if resume and os.path.exists(MODEL_PATH):
        print(" Resuming from previous model...")
        model.load_state_dict(torch.load(MODEL_PATH))

    print(" Training started...")
    score, state = train(model, train_loader, val_loader, optimizer)

    old_score = -1.0
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            old_score = json.load(f).get('score', -1.0)

    print(f" New score: {score*100:.2f}% | Previous best: {old_score*100:.2f}%")
    if score > old_score:
        print(" New model is better. Saving...")
        torch.save(state, MODEL_PATH)
        save_log(score)
    else:
        print(" New model is worse. Keeping previous.")

    print(" Done.")

if __name__ == '__main__':
    main()