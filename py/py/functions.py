import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
def load_star_data(path: str = "data/star_classification.csv") -> pd.DataFrame:
    """Load the SDSS star/galaxy/QSO classification dataset from CSV."""
    return pd.read_csv(path)

