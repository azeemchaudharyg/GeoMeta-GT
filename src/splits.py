import numpy as np
from sklearn.model_selection import train_test_split


def create_id_splits(metadata_processed, test_size=0.2, val_size=0.1, random_state=42):
    """
    Create train/val/test splits based on image IDs.

    Parameters
    ----------
    metadata_processed : pd.DataFrame
        Must contain column 'isic_id'

    test_size : float
        Fraction for test set

    val_size : float
        Fraction of training set used for validation

    random_state : int
        Reproducibility seed

    Returns
    -------
    train_ids, val_ids, test_ids
    """

    all_ids = metadata_processed["isic_id"].values

    train_ids, test_ids = train_test_split(all_ids, test_size=test_size, random_state=random_state, shuffle=True)

    train_ids, val_ids = train_test_split(train_ids, test_size=val_size, random_state=random_state, shuffle=True)

    train_ids = set(train_ids)
    val_ids = set(val_ids)
    test_ids = set(test_ids)

    print(
        f"Train: {len(train_ids)} | "
        f"Val: {len(val_ids)} | "
        f"Test: {len(test_ids)}"
    )

    return train_ids, val_ids, test_ids