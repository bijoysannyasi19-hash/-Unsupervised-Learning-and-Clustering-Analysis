import os
import pandas as pd
import requests

def download_data(url, dest_path):
    """Downloads the dataset if it doesn't exist."""
    if not os.path.exists(dest_path):
        print(f"Downloading data from {url}...")
        response = requests.get(url, verify=False)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("Data already exists. Skipping download.")

def load_data(filepath):
    """Loads the dataset into a pandas DataFrame."""
    return pd.read_csv(filepath)

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/TrainingByPackt/Data-Science-with-Python/master/Chapter01/Data/Wholesale%20customers%20data.csv"
    dest = "data/wholesale_customers.csv"
    download_data(url, dest)
    df = load_data(dest)
    print(df.head())
