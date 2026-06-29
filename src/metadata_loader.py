import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_metadata(metadata_path, data_dict):
    """
    Load and preprocess metadata.

    Parameters
    ----------
    metadata_path : str
        Path to metadata csv

    data_dict : dict
        Loaded pkl feature dictionary

    Returns
    -------
    metadata_processed : pd.DataFrame

    metadata_transformer : ColumnTransformer

    metadata_feature_dim : int
    """

    print(f"Loading metadata from {metadata_path}")

    metadata = pd.read_csv(metadata_path, sep=None, engine="python")

    # ----------------------------------
    # Normalize column names
    # ----------------------------------
    metadata.columns = (metadata.columns.str.lower().str.strip())

    # ----------------------------------
    # Remove label columns
    # ----------------------------------
    class_labels = ["lesion_id", "patient_id", "label", "target", "benign_malignant", "diagnosis", "diagnosis_1", "diagnosis_2", "diagnosis_3"]

    metadata = metadata.drop(
        columns=[c for c in class_labels if c in metadata.columns],
        errors="ignore"
    )

    # ----------------------------------
    # Rename columns
    # ----------------------------------
    rename_map = {"img_id": "isic_id"}

    metadata.rename(
        columns={k: v for k, v in rename_map.items() if k in metadata.columns},
        inplace=True
    )

    # ----------------------------------
    # Check ID column
    # ----------------------------------
    if "isic_id" not in metadata.columns:
        raise ValueError("Metadata file must contain 'isic_id' column")

    # ----------------------------------
    # Normalize image IDs
    # ----------------------------------
    metadata["isic_id"] = (
        metadata["isic_id"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(".jpg", "", regex=False)
        .str.replace(".png", "", regex=False)
    )

    # ----------------------------------
    # Normalize pkl keys
    # ----------------------------------
    pkl_ids = [k.split(".")[0].strip().lower()
        for k in data_dict.keys()
    ]

    common_ids = (set(metadata["isic_id"]) & set(pkl_ids))

    print(f"Matched {len(common_ids)} " f"images with metadata")

    metadata = metadata[
        metadata["isic_id"].isin(common_ids)
    ].reset_index(drop=True)

    # ----------------------------------
    # Identify column types
    # ----------------------------------
    numeric_cols = metadata.select_dtypes(include=[np.number]).columns.tolist()

    categorical_cols = metadata.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()

    categorical_cols = [c for c in categorical_cols if c != "isic_id"]

    # ----------------------------------
    # Missing values
    # ----------------------------------
    if len(numeric_cols) > 0:

        metadata[numeric_cols] = (
            metadata[numeric_cols]
            .fillna(
                metadata[numeric_cols].mean()
            )
        )

    for col in categorical_cols:
        metadata[col] = (
            metadata[col]
            .fillna("unknown")
        )

    # ----------------------------------
    # Preprocessing pipeline
    # ----------------------------------
    transformer = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols
            )])

    metadata_features = transformer.fit_transform(metadata.drop(columns=["isic_id"]))

    feature_names = (transformer.get_feature_names_out())

    metadata_processed = pd.DataFrame(metadata_features, columns=feature_names)

    metadata_processed["isic_id"] = (metadata["isic_id"].values)

    metadata_feature_dim = (metadata_processed.shape[1] - 1)

    print(f"Metadata shape: "
        f"{metadata_processed.shape}")

    print(f"Metadata feature dimension: "
        f"{metadata_feature_dim}")

    return (metadata_processed, transformer, metadata_feature_dim)


def get_metadata_row(metadata_processed, image_id):
    """
    Retrieve one image metadata row.
    """

    image_id = (
        image_id
        .replace(".jpg", "")
        .replace(".png", "")
        .strip()
        .lower()
    )

    row = metadata_processed[metadata_processed["isic_id"] == image_id]

    if row.empty:
        return None

    return row