import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from binance.client import Client
import time

BINANCE_API_KEY='UjLOE2csyHLsjyqJFCBBX2LXQSjDtfORW3oJUuicXeSDaSesPdnaVIwwipvJ2G6O'
BINANCE_API_SECRET='9FRVV24DIVu9VqsWfOdsjIP3LIDEVtzjtPcTKtUw0GK1bDr3VPNUmvLR8QLt5Tx2'

PAIR = 'TRXUSDT'
DATA_FILE = 'dataset.parquet'

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

# Завантаження klines даних

def fetch_klines(pair, interval, start_str, end_str=None):
    limit = 1000
    klines = []

    while True:
        print(f"\rFetching data from {pd.to_datetime(start_str, unit='ms', utc=True)}...", end='', flush=True)
        batch = client.get_historical_klines(pair, interval, start_str, end_str, limit=limit)
        if not batch:
            break

        klines.extend(batch)
        last_open_time = batch[-1][0]
        start_str = last_open_time + 1

        if len(batch) < limit:
            break

    print()
    return klines


def prepare_data():
    print(f"[{datetime.now(timezone.utc)}] Завантаження історичних даних...")

    if os.path.exists(DATA_FILE):
        df_existing = pd.read_parquet(DATA_FILE)
        start_time = df_existing['timestamp'].max() + timedelta(hours=1)
    else:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=5*365)

    h1_data = fetch_klines(PAIR, Client.KLINE_INTERVAL_1HOUR, int(start_time.timestamp()*1000))
    d1_data = fetch_klines(PAIR, Client.KLINE_INTERVAL_1DAY, int((start_time - timedelta(days=100)).timestamp()*1000))

    h1_df = pd.DataFrame(h1_data, columns=[
        'open_time', 'open', 'high', 'low', 'close',
        'drop1', 'drop2', 'drop3', 'drop4', 'drop5', 'drop6', 'drop7'
    ]).drop(columns=['drop1', 'drop2', 'drop3', 'drop4', 'drop5', 'drop6', 'drop7'])

    d1_df = pd.DataFrame(d1_data, columns=[
        'open_time', 'd1_open', 'd1_high', 'd1_low', 'd1_close', 'd1_volume',
        'drop1', 'drop2', 'drop3', 'drop4', 'drop5', 'drop6'
    ]).drop(columns=['drop1', 'drop2', 'drop3', 'drop4', 'drop5', 'drop6'])

    h1_df['open_time'] = pd.to_datetime(h1_df['open_time'], unit='ms', utc=True)
    d1_df['open_time'] = pd.to_datetime(d1_df['open_time'], unit='ms', utc=True)

    h1_df.rename(columns={'open_time': 'timestamp'}, inplace=True)

    # EMA для H1
    for span in [8, 24, 168]:
        h1_df[f'ema_{span}'] = h1_df['close'].astype(float).ewm(span=span).mean()

    # SMA для D1
    for period in [7, 30, 90]:
        d1_df[f'sma_{period}'] = d1_df['d1_close'].astype(float).rolling(window=period).mean()

    d1_df['open_time'] += timedelta(days=1)
    d1_df_resampled = d1_df.set_index('open_time').resample('1h').ffill()

    df = h1_df.set_index('timestamp').join(d1_df_resampled, how='left').reset_index()

    # Додавання циклічних фіч
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.weekday / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.weekday / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.month / 12)

    # Додавання future_high та future_low
    df['future_high'] = df['high'].shift(-1)
    df['future_low'] = df['low'].shift(-1)
    df = df[:-1]  # видаляємо останній запис, де вони відсутні

    if os.path.exists(DATA_FILE):
        df_existing = pd.read_parquet(DATA_FILE)
        df = pd.concat([df_existing, df]).drop_duplicates(subset=['timestamp']).reset_index(drop=True)

    # Збереження
    df.to_parquet(DATA_FILE)
    print(f"[{datetime.now(timezone.utc)}] Дані збережені в {DATA_FILE}")


while True:
    try:
        prepare_data()
    except Exception as e:
        print(f"[{datetime.now(timezone.utc)}] Виникла помилка: {e}")

    next_run = datetime.now(timezone.utc).replace(minute=3, second=0, microsecond=0) + timedelta(hours=1)
    sleep_seconds = (next_run - datetime.now(timezone.utc)).total_seconds()
    print(f"[{datetime.now(timezone.utc)}] Засинаємо на {sleep_seconds/60:.2f} хвилин...")
    time.sleep(sleep_seconds)
