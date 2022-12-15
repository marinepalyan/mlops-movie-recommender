#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
df = pd.read_csv("../data/cleaned_data.csv")


# ### Prediction model

# In[3]:


from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer
import numpy as np

# #### Regression

# In[30]:


X, y = df.drop(['name','gross', 'year', 'score', 'runtime'], axis = 1), df['gross'] # removed low correlated ones


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[32]:


num = list(X_train.select_dtypes('number').columns)
cat = list(X_train.select_dtypes('category').columns)

model_scale = MinMaxScaler()
model_encoder = OneHotEncoder()
transformer = ColumnTransformer([('num',model_scale,num),
                                 ('cat',model_encoder,cat)])

mod_ada_2 = AdaBoostRegressor(ExtraTreeRegressor()) # base_estimator=None by default
model_combined = make_pipeline(transformer, mod_ada_2)

n_estimators = [int(x) for x in np.linspace(start = 1, stop = 30, num = 1)]
parameters = {'adaboostregressor__n_estimators': n_estimators
             }
gridCV= GridSearchCV(mod_ada_2, parameters, cv=10)
model_combined_optimal = GridSearchCV(model_combined,
          param_grid = parameters, n_jobs = -1, cv = 10, return_train_score=True)
model_combined_optimal = model_combined_optimal.fit(X_train, y_train)
best_model_extra = model_combined_optimal.best_estimator_
best_model_extra


# In[33]:


best_model_extra.score(X_train,y_train)


# In[34]:


best_model_extra.score(X_test,y_test)


# #### Ridge regression

# In[35]:


model_scale = MinMaxScaler()
model_ridge = Ridge()
model_encoder = OneHotEncoder()
num = list(X_train.select_dtypes('number').columns)
cat = list(X_train.select_dtypes('category').columns)
transformer = ColumnTransformer([('num',model_scale,num),
                                 ('cat',model_encoder,cat)])
model_combined = make_pipeline(transformer, model_ridge)
alpha_grid = np.linspace(0.0001, 50, 50)
model_combined_optimal = GridSearchCV(model_combined, 
                             param_grid = {'ridge__alpha': np.linspace(0.0001, 50, 50)})
model_combined_optimal.fit(X_train,y_train)


# In[36]:


model_combined_optimal.best_params_


# In[37]:


model_combined_optimal.score(X_train,y_train)


# In[38]:


model_combined_optimal.score(X_test,y_test)


# #### Ridge regression reduced params

# In[39]:


X, y = df.drop(['name','gross', 'year', 'score', 'runtime', 'writer'], axis = 1), df['gross'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[40]:


model_scale = MinMaxScaler()
model_ridge = Ridge()
model_encoder = OneHotEncoder()
num = list(X_train.select_dtypes('number').columns)
cat = list(X_train.select_dtypes('category').columns)
transformer = ColumnTransformer([('num',model_scale,num),
                                 ('cat',model_encoder,cat)])
model_combined = make_pipeline(transformer, model_ridge)
alpha_grid = np.linspace(0.0001, 50, 50)
model_combined_optimal = GridSearchCV(model_combined, 
                             param_grid = {'ridge__alpha': np.linspace(0.0001, 1, 50)})
model_combined_optimal.fit(X_train,y_train)


# In[41]:


model_combined_optimal.best_params_


# In[42]:


model_combined_optimal.score(X_train,y_train)


# In[43]:


model_combined_optimal.score(X_test,y_test)


# #### Ridge for 'score' prediction

# In[7]:


X, y = df.drop(['name', 'year', 'score'], axis = 1), df['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[8]:


model_scale = MinMaxScaler(clip=False)
model_ridge = Ridge()
model_encoder = OneHotEncoder()
num = list(X_train.select_dtypes('number').columns)
cat = list(X_train.select_dtypes('category').columns)
transformer = ColumnTransformer([('num',model_scale,num),
                                 ('cat',model_encoder,cat)])
model_combined = make_pipeline(transformer, model_ridge)
alpha_grid = np.linspace(0.0001, 50, 50)
model_combined_optimal = GridSearchCV(model_combined, 
                             param_grid = {'ridge__alpha': np.linspace(0.0001, 50, 50)})
model_combined_optimal.fit(X_train,y_train)


# In[9]:


model_combined_optimal.best_params_


# In[10]:


model_combined_optimal.score(X_train,y_train)


# In[11]:


model_combined_optimal.score(X_test,y_test)


# In[ ]:

# import pickle
# with open('model.pkl', 'wb') as f:
#     pickle.dump(model_combined_optimal.best_estimator_, f)

import joblib
joblib.dump(model_combined_optimal.best_estimator_, 'model.pkl', compress=1)

