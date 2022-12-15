import pytest

import warnings
from src.movie_predictor.models.model_loading import load_model
from src.movie_predictor.data.load_data import load_data
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_prediction(model):
    data = load_data()
    X = data.drop(['name', 'gross', 'year', 'score', 'runtime', 'writer'], axis=1)
    pred = model.predict(X)
