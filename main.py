import csv
import time
from datetime import datetime, timedelta

import pandas as pd
from coinbase.rest import RESTClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class CryptoDataLoader:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path

    def load_data(self):
        data = pd.read_csv(self.csv_file_path)
        X = data.drop(columns=['close']).values
        y = data['close'].values
        return X, y


class CryptoTrader:
    def __init__(self, key_file):
        self.client = RESTClient(key_file=key_file)

    def get_candles_data(self, product_id, start, end, granularity):
        data = self.client.get_candles(product_id=product_id, start=start, end=end, granularity=granularity)
        candles = data['candles']

        csv_filename = f"{product_id}-{datetime.utcfromtimestamp(end).strftime('%Y-%m-%d')}-data.csv"
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=candles[0].keys())
            writer.writeheader()
            writer.writerows(candles)

        return csv_filename


class NeuralNetworkTrader:
    def __init__(self, input_shape):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)),  # Pass input_shape as a tuple
            Dense(32, activation='relu'),
            Dense(1)  # Output layer (1 neuron for regression)
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    def train_model(self, X_train_scaled, y_train):
        print("Training the neural network model...")
        history = self.model.fit(X_train_scaled, y_train, epochs=300, batch_size=32, validation_split=0.2)
        print("Training completed.")

    def evaluate_model(self, X_test_scaled, y_test):
        print("Evaluating the neural network model...")
        mse = self.model.evaluate(X_test_scaled, y_test)
        rmse = np.sqrt(mse)  # Calculate root mean squared error
        print(f'Root Mean Squared Error (RMSE): {rmse:.6f}')
        print("RMSE represents the average deviation of the predicted values from the actual values.")
        print("A lower RMSE indicates better performance of the model.")
        print("Evaluation completed.")

    def save_model(self, file_path):
        print(f"Saving the trained model to {file_path}...")
        self.model.save(file_path)
        print("Model saved successfully.")


def main():
    product = input('Enter product name: ')
    end_timestamp = int(datetime.now().timestamp())

    # Calculate the start timestamp (300 days earlier)
    start_timestamp = int((datetime.now() - timedelta(days=299)).timestamp())

    # Call the get_candles_data method with the calculated timestamps
    crypto_trader = CryptoTrader(key_file="coinbase_cloud_api_key.json")
    crypto = crypto_trader.get_candles_data(product_id=product, start=start_timestamp, end=end_timestamp,
                                            granularity="ONE_DAY")

    # Load and preprocess the data
    data_loader = CryptoDataLoader(crypto)
    x, y = data_loader.load_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Initialize and train the neural network model
    input_shape = x_train_scaled.shape[1]  # Get the shape of preprocessed input data
    trader = NeuralNetworkTrader(input_shape=input_shape)  # Pass input_shape argument
    trader.train_model(x_train_scaled, y_train)

    # Evaluate and save the model
    trader.evaluate_model(x_test_scaled, y_test)
    trader.save_model(f'{product}_neural_network_trader.keras')


if __name__ == '__main__':
    main()
