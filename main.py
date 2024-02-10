import csv
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
        return data['candles']


class NeuralNetworkTrader:
    def __init__(self, input_shape):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)),  # Pass input_shape as a tuple
            Dense(32, activation='relu'),
            Dense(1)  # Output layer (1 neuron for regression)
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    def train_model(self, X_train_scaled, y_train):
        history = self.model.fit(X_train_scaled, y_train, epochs=300, batch_size=32, validation_split=0.2)

    def evaluate_model(self, X_test_scaled, y_test):
        mse = self.model.evaluate(X_test_scaled, y_test)
        print(f'Mean Squared Error: {mse}')

    def save_model(self, file_path):
        self.model.save(file_path)


def main():
    # Load and preprocess the data
    data_loader = CryptoDataLoader('SHIB_data.csv')
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
    trader.save_model('neural_network_trader.keras')


if __name__ == '__main__':
    main()
