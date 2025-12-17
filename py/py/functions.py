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
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split df into feature matrix X and target vector y.
    Applies basic column dropping and one-hot encoding for categorical features.
    """
    df = df.copy()

    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    y = df[target_col]
    X = df.drop(columns=[target_col])

    if categorical_cols:
        X = pd.get_dummies(X, columns=[c for c in categorical_cols if c in X.columns], drop_first=False)

    return X, y

from sklearn.model_selection import train_test_split

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train/val/test. val_size is a fraction of the training split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

from sklearn.preprocessing import StandardScaler

def scale_features(
    X_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame
) -> Tple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardise features using statistics from X_train only.
    Returns numpy arrays + the fitted scaler.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scale.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return x_train_s, X_val_s, X_test_s, scaler
