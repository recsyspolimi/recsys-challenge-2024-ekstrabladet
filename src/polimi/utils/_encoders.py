from sklearn.preprocessing import TargetEncoder
import numpy as np
from ebrec.utils._behaviors import (
    create_binary_labels_column,
)
import polars as pl


def build_target_encoder_features(df_features: pl.DataFrame, behaviors: pl.DataFrame, cols=['user_id'], target='labels') -> pl.DataFrame:
    df = behaviors.pipe(create_binary_labels_column, shuffle=True, seed=123)\
        .select(cols + [target])\
        .explode(target)\
        .sort(cols)

    encoder = build_target_encoder(df.select(cols).to_numpy(), df['labels'].to_numpy())
    X_enc = encoder.transform(df_features.select(cols).to_numpy())
    df_features = df_features.with_columns(pl.Series(X_enc[:, i]).alias(f'{col}_target_encoded') for i, col in enumerate(cols))
    
    return df_features, encoder

def build_target_encoder(X:np.ndarray, y:np.ndarray, smooth='auto', target_type='binary', cv=5):
    encoder = TargetEncoder(smooth=smooth, target_type=target_type, cv=cv).fit(X, y)
    return encoder