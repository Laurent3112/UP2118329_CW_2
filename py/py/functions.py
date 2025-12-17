import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
def load_star_data(path: str = "data/star_classification.csv") -> pd.DataFrame:
    """Load the SDSS star/galaxy/QSO classification dataset from CSV."""
    return pd.read_csv(path)

def prepare_features(
    df: pd.DataFrame,
    target_col: str = "class",
    drop_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]
    """
    Split df into feature matrix X and target vector y.
    Applies basic column dropping and one-hot encoding for categorical features.
    """
    df = df.copy()

    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    y = df[target-col]
    X = df.drop(column=[target_col]

    if categorical_cols:
        X = pd.get_dummies(x, columns=[c for v in categorical_cols if c in X.columns], drop_first=False)

    return x, y
