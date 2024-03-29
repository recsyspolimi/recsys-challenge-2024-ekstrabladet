import numpy as np
from typing_extensions import List


def cosine_similarity(x: List[float], y: List[float]):
    x = np.array(x)
    y = np.array(y)
    normalization = np.linalg.norm(x, 2) * np.linalg.norm(y, 2)
    return np.dot(x, y) / normalization if normalization > 0 else 0