import numpy as np
from typing import Tuple

def create_dataset_symmetric_2point() -> Tuple[np.ndarray, np.ndarray]:
    """
    Default 2-point dataset:
      x = [-1, 1], y = [-1, 1]
    """
    x = np.array([-1.0, 1.0], dtype=float)
    y = np.array([-1.0, 1.0], dtype=float)
    return x, y