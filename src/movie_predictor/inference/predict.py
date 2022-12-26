import numpy as np

from src.movie_predictor.models.model_loading import load_model, load_scaler, load_encoder


def predict(input_data) -> dict:
    """Make a prediction using a saved model pipeline."""
    model = load_model()
    scaler = load_scaler()
    encoder = load_encoder()
    # separate cat and num columns and apply correct methods
    num = list(input_data.select_dtypes('number').columns)
    cat = list(input_data.select_dtypes('category').columns)
    num_scaled = scaler.transform(input_data[num])
    cat_encoded = encoder.transform(input_data[cat]).toarray()
    X = np.concatenate([num_scaled, cat_encoded], axis=1)
    # make prediction
    prediction = model.predict(X)
    return prediction
