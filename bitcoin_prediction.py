import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
import os

class BitcoinPricePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.features = ['Close', 'Returns', 'MA7', 'MA30', 'Volatility', 'RSI', 'MACD']
        self.model_file = 'bitcoin_model.joblib'
        self.scaler_file = 'scaler.joblib'

    def fetch_bitcoin_data(self, start_date='2015-01-01'):
        """Fetch historical Bitcoin data from the specified start date until today"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        print(f"Fetching Bitcoin data from {start_date} to {end_date}...")
        return yf.download('BTC-USD', start=start_date, end=end_date)

    def calculate_rsi(self, data, periods=14):
        """Calculate Relative Strength Index"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, data, short_window=12, long_window=26):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        short_ema = data['Close'].ewm(span=short_window).mean()
        long_ema = data['Close'].ewm(span=long_window).mean()
        return short_ema - long_ema

    def create_features(self, df):
        """Create technical indicators as features"""
        df['Returns'] = df['Close'].pct_change()
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA30'] = df['Close'].rolling(window=30).mean()
        df['Volatility'] = df['Returns'].rolling(window=30).std()
        df['RSI'] = self.calculate_rsi(df)
        df['MACD'] = self.calculate_macd(df)
        
        # Create target variables for different prediction horizons
        df['Target_1d'] = df['Close'].shift(-1)
        df['Target_7d'] = df['Close'].shift(-7)
        df['Target_30d'] = df['Close'].shift(-30)
        
        return df.dropna()

    def prepare_data(self, df, target_column='Target_1d'):
        """Prepare data for modeling"""
        X = df[self.features]
        y = df[target_column]
        
        X_scaled = self.scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
        """Train the model with hyperparameter tuning"""
        print("\nStarting model training...")
        print(f"Training data shape: {X_train.shape}")
        
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        
        print("\nPerforming grid search with cross-validation...")
        knn = KNeighborsRegressor()
        grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        print(f"\nBest parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation score: {-grid_search.best_score_:.2f} MSE")
        
        print("\nSaving model and scaler...")
        joblib.dump(self.model, self.model_file)
        joblib.dump(self.scaler, self.scaler_file)
        print("Model and scaler saved successfully!")

    def load_model(self):
        """Load pre-trained model if available"""
        if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
            self.model = joblib.load(self.model_file)
            self.scaler = joblib.load(self.scaler_file)
            return True
        return False

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"Mean Squared Error: ${mse:,.2f}")
        print(f"Mean Absolute Error: ${mae:,.2f}")
        print(f"R-squared Score: {r2:.4f}")
        
        return y_test, y_pred

    def plot_results(self, y_test, y_pred, title):
        """Plot actual vs predicted prices"""
        plt.figure(figsize=(12, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.title(title)
        plt.tight_layout()
        plt.savefig('prediction_results.png')
        plt.close()

    def predict_future_price(self, days_ahead=1):
        """Predict future Bitcoin price"""
        try:
            # Get latest data with a wider date range to ensure we have enough data
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            latest_data = self.fetch_bitcoin_data(start_date=start_date)
            
            if latest_data.empty:
                raise ValueError("No data received from Yahoo Finance")
            
            df = self.create_features(latest_data)
            
            if df.empty:
                raise ValueError("No features could be created from the data")
            
            # Get the last complete row of data
            latest_features = df[self.features].dropna().iloc[-1:]
            
            if latest_features.empty:
                raise ValueError("No valid features available for prediction")
            
            # Scale the features
            scaled_features = self.scaler.transform(latest_features)
            
            # Make prediction
            predicted_price = float(self.model.predict(scaled_features)[0])
            current_price = float(df['Close'].iloc[-1])
            
            return current_price, predicted_price
            
        except Exception as e:
            print(f"\nError during prediction: {str(e)}")
            print("Please try retraining the model with latest data (option 2)")
            return None, None

def main():
    predictor = BitcoinPricePredictor()
    
    # Check if we have a saved model
    if not predictor.load_model():
        print("Training new model...")
        # Fetch and prepare data
        btc_data = predictor.fetch_bitcoin_data()
        df = predictor.create_features(btc_data)
        
        # Train model
        X_train, X_test, y_train, y_test = predictor.prepare_data(df)
        predictor.train_model(X_train, y_train)
        
        # Evaluate model
        y_test, y_pred = predictor.evaluate_model(X_test, y_test)
        predictor.plot_results(y_test, y_pred, 'Bitcoin Price Prediction: Actual vs Predicted')
    else:
        print("Loaded pre-trained model")

    while True:
        print("\nBitcoin Price Prediction Menu:")
        print("1. Predict next day's price")
        print("2. Retrain model with latest data")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            current_price, predicted_price = predictor.predict_future_price()
            if current_price is not None and predicted_price is not None:
                print(f"\nCurrent Bitcoin Price: ${current_price:,.2f}")
                print(f"Predicted Next Day Price: ${predicted_price:,.2f}")
                print(f"Predicted Change: {((predicted_price - current_price) / current_price * 100):,.2f}%")
        
        elif choice == '2':
            print("Retraining model with latest data...")
            btc_data = predictor.fetch_bitcoin_data()
            df = predictor.create_features(btc_data)
            X_train, X_test, y_train, y_test = predictor.prepare_data(df)
            predictor.train_model(X_train, y_train)
            y_test, y_pred = predictor.evaluate_model(X_test, y_test)
            predictor.plot_results(y_test, y_pred, 'Bitcoin Price Prediction: Actual vs Predicted (Retrained)')
        
        elif choice == '3':
            print("Thank you for using the Bitcoin Price Predictor!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
