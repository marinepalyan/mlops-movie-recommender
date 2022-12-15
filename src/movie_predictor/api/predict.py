import logging
from typing import Union, Dict

import pandas as pd
import pickle

_logger = logging.getLogger(__name__)
model = pickle.load(open('models/model.pkl', 'rb'))
def make_prediction(*, input_data: Union[pd.DataFrame, Dict],) -> Dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)


    predictions = model.predict(
        X=data
        )

    results = {"predictions": predictions}

    return results
