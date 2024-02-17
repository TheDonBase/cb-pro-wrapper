from datetime import datetime, timedelta
from src.crypto_trader import CryptoTrader


def main():
    # Define the path to your API key file and data file
    key_file = "coinbase_cloud_api_key.json"

    # Initialize the CryptoTrader
    crypto_trader = CryptoTrader(key_file=key_file)

    # Calculate start and end timestamps
    end_time = int(datetime.now().timestamp())

    # Calculate the start timestamp (300 days earlier)
    start_time = int((datetime.now() - timedelta(days=299)).timestamp())

    # Specify product ID and granularity
    product_id = "BTC-USD"
    granularity = "ONE_DAY"  # 1 hour

    # Get candles data
    csv_filename = crypto_trader.get_candles_data(product_id=product_id, start=start_time, end=end_time,
                                                  granularity=granularity)

    print(f"Candles data saved to {csv_filename}")


if __name__ == "__main__":
    main()
