import pytest

import warnings
from movie_predictor.models.model_loading import load_model
from movie_predictor.data.load_data import load_data

from movie_predictor.inference.predict import predict
from movie_predictor.api.config import config

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_prediction(model):
    data = load_data()
    X = data.drop(columns=config.model_config.drop_features, axis='columns')
    pred = predict(input_data=X.iloc[0:1])
    assert pred[0] == pytest.approx(8.93, 0.001)
