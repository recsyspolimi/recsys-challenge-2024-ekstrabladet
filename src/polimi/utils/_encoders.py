from sklearn.preprocessing import TargetEncoder
import numpy as np

def build_target_encoder(X:np.ndarray, y:np.ndarray, smooth='auto', target_type='binary', cv=5):
    encoder = TargetEncoder(smooth=smooth, target_type=target_type, cv=cv).fit(X, y)
    return encoder