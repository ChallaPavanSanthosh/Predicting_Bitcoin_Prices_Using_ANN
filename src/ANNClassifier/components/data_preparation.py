import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import ta  # Technical Analysis library (ta-lib alternative)
import pickle

class DataPreparation:
    def __init__(self, file_path, sample_size=10000, target_column='Close'):
        self.file_path = file_path
        self.sample_size = sample_size
        self.target_column = target_column

    def load_data(self):
        df = pd.read_csv(self.file_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.set_index('Timestamp', inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        return df

    def feature_engineering(self, df):
        df['Volume_Lag1'] = df['Volume'].shift(1)
        df['Close_MA7'] = df['Close'].rolling(window=7).mean()
        df['Close_PctChange'] = df['Close'].pct_change()
        df['High_Low_Range'] = df['High'] - df['Low']
        df['Price_Change'] = df['Close'] - df['Open']
        df['Volatility'] = df['Close'].rolling(window=7).std()

        # Add indicators
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close']).rsi()
        df['MACD'] = ta.trend.MACD(close=df['Close']).macd()
        bb = ta.volatility.BollingerBands(close=df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()

        df.dropna(inplace=True)
        return df

    def prepare(self):
        df = self.load_data()
        df = self.feature_engineering(df)
        if self.sample_size:
            df = df.iloc[:self.sample_size]

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

        X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.3, shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

        # ✅ Save train/val data to .pkl
        os.makedirs("artifacts/data_preparation", exist_ok=True)
        with open("artifacts/data_preparation/train_val_data.pkl", "wb") as f:
            pickle.dump({
                "X_train": X_train,
                "X_val": X_val,
                "y_train": y_train,
                "y_val": y_val
            }, f)

        return X_train, X_val, X_test, y_train, y_val, y_test, scaler_y, X.shape[1]
