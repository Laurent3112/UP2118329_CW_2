import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
def load_star_data(path="data/star_classification.csv"):
    import pandas as pd
    return pd.read_csv(path)
