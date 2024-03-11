import os
import logging

import pandas as pd
import datasets
import numpy as np

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_data(data_dir_path: str, parts_num: int = 2):

    sbol_path = os.path.join(data_dir_path, "sbol")
    smm_path = os.path.join(data_dir_path, "smm")

    features_count = 1345
    sample = 10_000
    seed = 22

    # preparing labels
    sbol_labels = pd.read_parquet(os.path.join(sbol_path, "sbol_multilabels.parquet"))
    sbol_labels = sbol_labels.iloc[:sample]

    users_train, users_test = train_test_split(
        sbol_labels[["user_id"]], shuffle=True, random_state=seed, test_size=0.15
    )

    train_labels_df = users_train.merge(sbol_labels, on="user_id", how="left")
    test_labels_df = users_test.merge(sbol_labels, on="user_id", how="left")

    train_ds = datasets.Dataset.from_dict(
        {
            "user_id": train_labels_df["user_id"],
            "labels": train_labels_df["labels"],
        }
    )

    test_ds = datasets.Dataset.from_dict(
        {
            "user_id": test_labels_df["user_id"],
            "labels": test_labels_df["labels"],
        }
    )

    ds = datasets.DatasetDict({'train_train': train_ds, 'train_val': test_ds})

    ds.save_to_disk(
        os.path.join(data_dir_path, f"vfl_multilabel_sber_sample{sample}_parts{parts_num}", f"master_part")
    )

    # preparing sbol user features
    sbol_user_features = pd.read_parquet(os.path.join(sbol_path, "user_features.parquet"))
    sbol_user_features.fillna(0, inplace=True)

    # filtering
    sbol_user_features = sbol_labels[["user_id"]].merge(sbol_user_features, on="user_id", how="left")

    sbol_user_features["features_part_0"] = sbol_user_features[[f'feature_{x}' for x in range(features_count)]].apply(
        lambda x: list(x), axis=1)

    sbol_user_features = sbol_user_features[["user_id", "features_part_0"]]

    train_sbol_df = users_train.merge(sbol_user_features, on="user_id", how="left")
    test_sbol_df = users_test.merge(sbol_user_features, on="user_id", how="left")

    part = 0
    train_ds = datasets.Dataset.from_dict(
        {
            "user_id": train_sbol_df["user_id"],
            "features_part_0": train_sbol_df["features_part_0"],
        }
    )

    test_ds = datasets.Dataset.from_dict(
        {
            "user_id": test_sbol_df["user_id"],
            "features_part_0": test_sbol_df["features_part_0"],
        }
    )

    ds = datasets.DatasetDict({'train_train': train_ds, 'train_val': test_ds})

    ds.save_to_disk(
        os.path.join(data_dir_path, f"vfl_multilabel_sber_sample{sample}_parts{parts_num}", f"part_{part}")
    )

    # preparing smm user features

    smm_user_factors = pd.read_parquet(os.path.join(smm_path, "als_user_factors.parquet"))
    # filtering
    smm_user_factors = sbol_labels[["user_id"]].merge(smm_user_factors, on="user_id", how="inner")
    smm_user_factors.rename(columns={"user_factors": "features_part_1"}, inplace=True)

    train_smm_df = users_train.merge(smm_user_factors, on="user_id", how="left")
    test_smm_df = users_test.merge(smm_user_factors, on="user_id", how="left")

    train_smm_df.dropna(inplace=True)
    test_smm_df.dropna(inplace=True)

    part = 1
    train_ds = datasets.Dataset.from_dict(
        {
            "user_id": train_smm_df["user_id"],
            f"features_part_{part}": train_smm_df[f"features_part_{part}"],
        }
    )

    test_ds = datasets.Dataset.from_dict(
        {
            "user_id": test_smm_df["user_id"],
            f"features_part_{part}": test_smm_df[f"features_part_{part}"],
        }
    )

    ds = datasets.DatasetDict({'train_train': train_ds, 'train_val': test_ds})

    ds.save_to_disk(
        os.path.join(data_dir_path, f"vfl_multilabel_sber_sample{sample}_parts{parts_num}", f"part_{part}")
    )
