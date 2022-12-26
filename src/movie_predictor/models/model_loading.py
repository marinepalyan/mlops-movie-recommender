from pathlib import Path

import joblib


def load_model():
    # load from this folder
    model_path = Path(__file__).parent / "model.pkl"
    with open(model_path, 'rb') as f:
        return joblib.load(f)


def load_scaler():
    # load from this folder
    scaler_path = Path(__file__).parent / "scaler.pkl"
    with open(scaler_path, 'rb') as f:
        return joblib.load(f)


def load_encoder():
    # load from this folder
    encoder_path = Path(__file__).parent / "encoder.pkl"
    with open(encoder_path, 'rb') as f:
        return joblib.load(f)
