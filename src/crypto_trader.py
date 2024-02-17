import csv
import os
from datetime import datetime

from coinbase.rest import RESTClient

from src.neural_network import NeuralNetwork


class CryptoTrader:
    def __init__(self, key_file):
        self.client = RESTClient(key_file=key_file)
        self.neural_network = NeuralNetwork()

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
