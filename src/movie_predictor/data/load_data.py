import pandas as pd
from pathlib import Path


def load_data():
    data_path = Path(__file__).parent / "cleaned_data.csv"
    data = pd.read_csv(data_path)
    cat_cols = ['rating', 'genre', 'director', 'writer', 'star', 'country']
    for col in cat_cols:
        data[col] = data[col].astype('category')
    return data
