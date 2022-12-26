import pytest

import warnings
from src.movie_predictor.models.model_loading import load_model
from src.movie_predictor.data.load_data import load_data

from src.movie_predictor.inference.predict import predict

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_prediction(model):
    data = load_data()
    X = data.drop(columns=['name', 'year', 'score'], axis='columns')
    pred = predict(input_data=X.iloc[0:1])
    assert pred[0] == pytest.approx(8.93, 0.001)
