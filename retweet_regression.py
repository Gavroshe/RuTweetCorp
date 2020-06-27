import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, Ridge, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from imblearn.over_sampling import ADASYN

columns = ['id', 'tdate', 'tmane', 'ttext', 'ttype', 'trep', 'trtw', 'tfav', 'tstcount', 'tfol', 'tfrien', 'listcount']
pos = pd.read_csv('positive.csv', sep=';', names=columns)
neg = pd.read_csv('negative.csv', sep=';', names=columns)

data = pos.append(neg)

data_shape = data.loc[(data['trtw'] >= 0) & (~data['ttext'].str.contains('RT'))].sort_values(by=['trtw', 'tdate'])
data_shape.loc[data_shape['trtw'] == 2, 'trtw'] = 1

data = data_shape[['tmane', 'ttext', 'ttype', 'tstcount', 'tfol', 'tfrien', 'listcount']]
y = data_shape['trtw']

data['numwords'] = data['ttext'].apply(lambda x: len(x.split(" ")))
data['lentweet'] = data['ttext'].apply(lambda x: len(x))

ada = ADASYN(random_state=42)

preprocessor = ColumnTransformer(
        transformers=[('text', CountVectorizer(), 'ttext'), ('name', CountVectorizer(), 'tmane')])
data_trans = preprocessor.fit_transform(data)

svd = TruncatedSVD(n_components=50, n_iter=4, random_state=42)
X_features = svd.fit_transform(data_trans)

X_train, y_train = ada.fit_resample(X_features, y)

mae = make_scorer(mean_absolute_error)

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('regr', Lasso())
])

param_grid = [
    {
        'regr': [Lasso(), Ridge()],
        'regr__alpha': np.logspace(-4, 1, 6),
    },
    {
        'regr': [SGDRegressor()],
        'regr__alpha': np.logspace(-5, 0, 6),
        'regr__max_iter': [500, 1000],
    },

    {
        'regr': [RandomForestRegressor()],
        'regr__bootstrap': [False],
        'regr__max_depth': [10, 50],
        'regr__max_features': ['auto'],
        'regr__min_samples_leaf': [2, 4],
        'regr__min_samples_split': [2, 5],
        'regr__n_estimators': [200, 500]

    },

    {'regr': [XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                           max_depth=5, alpha=10, n_estimators=10)],
     'regr__min_child_weight': [0.5, 1],
     'regr__gamma': [5, 10],
     'regr__subsample': [0.7, 1.0],
     'regr__colsample_bytree': [0.5, 1.0],
     'regr__max_depth': [8, 12]
     }
]

grid = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1, verbose=10, scorer=mae)
grid.fit(X_train, y_train)

print("Best score: %0.6f" % grid.best_score_)
print("Best parameters set:")
best_parameters = grid.best_estimator_.get_params()
print(best_parameters)
