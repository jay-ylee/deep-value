from typing import Final
import numpy as np

AFTER_INCLUDE_ONLY: bool = True

BINS: Final[dict[str, list[float]]] = {
    '4': [-np.inf, -0.4, -0.1, -0.01, 0],
    '10': [-np.inf, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05, -0.01, 0],
}
