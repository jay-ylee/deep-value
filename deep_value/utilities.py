import os
import numpy as np
import pandas as pd

def get_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir