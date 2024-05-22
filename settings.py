from typing import Final
import numpy as np

AFTER_INCLUDE_ONLY: bool = True

BINS: Final[dict[str, list[float]]] = {
    '4(-0.3)': [-np.inf, -0.3, -0.1, -0.01, 0],
    '4(-0.4)': [-np.inf, -0.4, -0.1, -0.01, 0],
    '10(-0.7)': [-np.inf, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05, -0.01, 0],
}

ORANGES = [
    "rgb(245, 223, 204)",
    "rgb(245, 207, 179)",
    "rgb(245, 191, 153)",
    "rgb(245, 175, 128)",
    "rgb(245, 159, 102)",
    "rgb(245, 143, 77)",
    "rgb(245, 127, 51)",
    "rgb(245, 111, 25)",
    "rgb(245, 95, 0)",
    "rgb(245, 130, 32)"
]