import pandas as pd
from pathlib import Path


def load_data():
    data_path = Path(__file__).parent / "cleaned_data.csv"
    return pd.read_csv(data_path)
