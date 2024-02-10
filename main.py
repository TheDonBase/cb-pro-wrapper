import csv

from datetime import datetime, timedelta
import os
import pandas as pd
from coinbase.rest import RESTClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


class CryptoDataLoader:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path

    def load_data(self):
        # Load the CSV file
        data = pd.read_csv(self.csv_file_path)

        # Extract additional features from the 'start' column if needed (e.g., day of the week, month, etc.)

        # Define the input features (X) and the target variable (y)
        x = data[['start', 'volume']].values  # Features: 'start' (date) and 'volume'
        y = data['close'].values  # Target variable: 'close' price

        return x, y


class CryptoTrader:
    def __init__(self, key_file):
        self.client = RESTClient(key_file=key_file)

    def get_candles_data(self, product_id, start, end, granularity):
        data = self.client.get_candles(product_id=product_id, start=start, end=end, granularity=granularity)
        candles = data['candles']

        # Construct the file path within the "data" directory
        data_dir = "data"
        csv_filename = f"{data_dir}/{datetime.utcfromtimestamp(end).strftime('%Y-%m-%d')}/{product_id}-data.csv"

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

        # Write the data to the CSV file
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=candles[0].keys())
            writer.writeheader()
            writer.writerows(candles)

        return csv_filename


class NeuralNetworkTrader:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.scaler = StandardScaler()  # Initialize scaler
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')  # Change activation to 'linear' for regression
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

    def train_model(self, x_train, y_train, epochs=1000, batch_size=32):
        print("Training the neural network model...")
        # Fit the scaler to the training data
        self.scaler.fit(x_train)
        # Scale the training data
        x_train_scaled = self.scaler.transform(x_train)
        # Train the model
        history = self.model.fit(x_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                                 verbose=1)
        print("Training completed.")

    def evaluate_model(self, x_test, y_test):
        print("Evaluating the neural network model...")
        # Scale the test data
        x_test_scaled = self.scaler.transform(x_test)
        loss, mae = self.model.evaluate(x_test_scaled, y_test)
        print(f'Mean Absolute Error: {mae:.6f}')
        print("Evaluation completed.")

    def save_model(self, file_path):
        print(f"Saving the trained model to {file_path}...")
        self.model.save(file_path)
        print("Model saved successfully.")

    @staticmethod
    def predict_from_model(model_file_path, new_csv_file_path):
        # Load the saved Keras model
        model = tf.keras.models.load_model(model_file_path)

        # Initialize CryptoDataLoader with the new CSV file path
        data_loader = CryptoDataLoader(new_csv_file_path)

        # Load the new data from the CSV file
        x_new, y_new = data_loader.load_data()

        # Preprocess the new data
        scaler = StandardScaler()
        x_new_scaled = scaler.fit_transform(x_new)

        # Make predictions
        predictions = model.predict(x_new_scaled)
        wrong_predictions = 0
        correct_predictions = 0
        print("Predictions:")
        for i, prediction in enumerate(predictions):
            if prediction[0] > y_new[i]:
                direction = "higher"
            elif prediction[0] < y_new[i]:
                direction = "lower"
            else:
                direction = "equal"

            is_correct = direction == "higher" and prediction[0] > y_new[i]
            if is_correct:
                correct_predictions += 1
            else:
                wrong_predictions += 1

            print(
                f"Prediction: {prediction[0]:.6f}, Actual: {y_new[i]:.6f}, Direction: {direction}, Correct: {is_correct}")

        print(f"Correct predictions: {correct_predictions}\nWrong predictions: {wrong_predictions}")


def main():
    product = input('Enter product name: ')
    end_timestamp = int(datetime.now().timestamp())

    # Calculate the start timestamp (300 days earlier)
    start_timestamp = int((datetime.now() - timedelta(days=299)).timestamp())

    # Call the get_candles_data method with the calculated timestamps
    crypto_trader = CryptoTrader(key_file="coinbase_cloud_api_key.json")
    crypto = crypto_trader.get_candles_data(product_id=product, start=start_timestamp, end=end_timestamp,
                                            granularity="ONE_DAY")

    data_loader = CryptoDataLoader(crypto)
    x, y = data_loader.load_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Initialize and train the neural network model
    input_shape = x_train.shape[1]
    trader = NeuralNetworkTrader(input_shape=input_shape)
    trader.train_model(x_train, y_train)

    # Evaluate the model
    trader.evaluate_model(x_test, y_test)

    trader_model = f"models/{product}_neural_network_trader.keras"
    trader.save_model(trader_model)

    trader.predict_from_model(trader_model, crypto)


if __name__ == '__main__':
    main()
