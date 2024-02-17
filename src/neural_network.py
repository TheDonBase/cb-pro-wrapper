import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class NeuralNetwork:
    def __init__(self):
        self.data_file = None
        self.data = None
        self.data_scaled = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = None
        self.scaler = MinMaxScaler()

    def load_data(self):
        # Load the data from CSV file
        self.data = pd.read_csv(self.data_file)

    def preprocess_data(self):
        # Convert string columns to numeric
        self.data["start"] = pd.to_numeric(self.data["start"])
        self.data["low"] = pd.to_numeric(self.data["low"])
        self.data["high"] = pd.to_numeric(self.data["high"])
        self.data["open"] = pd.to_numeric(self.data["open"])
        self.data["close"] = pd.to_numeric(self.data["close"])
        self.data["volume"] = pd.to_numeric(self.data["volume"])

        # Select features and target variable
        features = ["high", "open", "volume"]
        target = "close"

        # Scale selected features
        self.data_scaled = self.scaler.fit_transform(self.data[features + [target]])

        # Split data into features and target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_scaled[:, :-1],
                                                                                self.data_scaled[:, -1], test_size=0.2,
                                                                                random_state=42)

    def build_model(self):
        # Define the model architecture
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),
            Dense(64, activation='relu'),
            Dense(1)
        ])

        # Compile the model
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def train_model(self, epochs=50, batch_size=32, validation_split=0.2):
        # Train the model
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size,
                       validation_split=validation_split)

    def evaluate_model(self):
        # Evaluate the model
        loss, mae = self.model.evaluate(self.X_test, self.y_test)
        print("Test Mean Absolute Error:", mae)

    def save_model(self, filename):
        # Save the trained model
        self.model.save(f"../models/{filename}")

    def predict(self, data):
        # Make predictions using the trained model
        scaled_data = self.scaler.transform(data)
        return self.model.predict(scaled_data)
