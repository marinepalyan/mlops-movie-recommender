import pytest
import warnings
from src.movie_predictor.models.model_loading import load_model
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_model_structure(model):
    assert len(model.named_steps) == 2


def test_scaler(model):
    assert model.named_steps['columntransformer'].transformers[0][0] == 'num'
    assert isinstance(model.named_steps['columntransformer'].transformers[0][1], MinMaxScaler)
    assert model.named_steps['columntransformer'].transformers[0][2] == ['votes', 'budget']


def test_encoder(model):
    assert model.named_steps['columntransformer'].transformers[1][0] == 'cat'
    assert isinstance(model.named_steps['columntransformer'].transformers[1][1], OneHotEncoder)
