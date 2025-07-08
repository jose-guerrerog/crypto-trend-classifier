import requests
import pandas as pd
import os

def get_binance_data(symbol="BTCUSDT", interval="1h", limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore'
    ])

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    df = df.astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'})

    return df

if __name__ == "__main__":
    df = get_binance_data()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/btcusdt_1h.csv", index=False)
    print("Saved data to data/btcusdt_1h.csv")