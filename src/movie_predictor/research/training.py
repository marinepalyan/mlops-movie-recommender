from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("../data/cleaned_data.csv")
X, y = df.drop(['name', 'year', 'score'], axis=1), df['score']

# Fix some column types
cat_cols = ['rating', 'genre', 'director', 'writer', 'star', 'country']
for col in cat_cols:
    X[col] = X[col].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

num = list(X_train.select_dtypes('number').columns)
cat = list(X_train.select_dtypes('category').columns)

scaler = MinMaxScaler()
train_num_scaled = scaler.fit_transform(X_train[num])
test_num_scaled = scaler.transform(X_test[num])

encoder = OneHotEncoder().fit(X[cat])
train_cat_encoded = encoder.transform(X_train[cat]).toarray()
X_train = np.concatenate([train_num_scaled, train_cat_encoded], axis=1)

model_ridge = Ridge()
alpha_grid = np.linspace(0.0001, 50, 50)
model_combined_optimal = GridSearchCV(model_ridge,
                                      param_grid={'alpha': np.linspace(0.0001, 1, 50)})
model_combined_optimal.fit(X_train, y_train)
print(model_combined_optimal.best_params_)
print(model_combined_optimal.score(X_train, y_train))

num = list(X_test.select_dtypes('number').columns)
cat = list(X_test.select_dtypes('category').columns)
test_cat_encoded = encoder.transform(X_test[cat]).toarray()
X_test = np.concatenate([test_num_scaled, test_cat_encoded], axis=1)

print(model_combined_optimal.score(X_test, y_test))

model_path = Path(__file__).parents[1] / "model.pkl"
with open(model_path, 'wb') as f:
    joblib.dump(model_combined_optimal, f)

scaler_path = Path(__file__).parents[1] / "scaler.pkl"
with open(scaler_path, 'wb') as f:
    joblib.dump(scaler, f)

encoder_path = Path(__file__).parents[1] / "encoder.pkl"
with open(encoder_path, 'wb') as f:
    joblib.dump(encoder, f)
