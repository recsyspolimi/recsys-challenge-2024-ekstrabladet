from rgf import FastRGFClassifier
from typing_extensions import List, Union
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np


class FastRGFClassifierWrapper:
    
    def __init__(self, categorical_features: List[str], **fast_rgf_params):
        self.categorical_features = categorical_features
        self.model = FastRGFClassifier(**fast_rgf_params)
        self.encoder = OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False)
        
    def fit(self, X_train: pd.DataFrame, y_train: Union[pd.Series, np.ndarray], **kwargs):
        X_train_categorical = self.encoder.fit_transform(X_train[self.categorical_features], y_train)
        X_train_numerical = X_train.drop(self.categorical_features).replace([-np.inf, np.inf], np.nan).fillna(0).values
        X_train_transformed = np.concatenate([X_train_numerical, X_train_categorical])
        self.model.fit(X_train_transformed, y_train)
        
    def predict(self, X: pd.DataFrame):
        X_categorical = self.encoder.transform(X[self.categorical_features])
        X_numerical = X.drop(self.categorical_features).replace([-np.inf, np.inf], np.nan).fillna(0).values
        X_transformed = np.concatenate([X_numerical, X_categorical])
        return self.model.predict(X_transformed)
    
    def predict_proba(self, X: pd.DataFrame):
        X_categorical = self.encoder.transform(X[self.categorical_features])
        X_numerical = X.drop(self.categorical_features).replace([-np.inf, np.inf], np.nan).fillna(0).values
        X_transformed = np.concatenate([X_numerical, X_categorical])
        return self.model.predict(X_transformed)
        