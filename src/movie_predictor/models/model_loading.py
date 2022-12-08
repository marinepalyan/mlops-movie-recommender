import pickle
from pathlib import Path


def load_model():
    # load from this folder
    model_path = Path(__file__).parent / "model.pkl"
    with open(model_path, 'rb') as f:
        return pickle.load(f)
