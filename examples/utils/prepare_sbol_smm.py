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

    sbol_labels = pd.read_parquet(os.path.join(sbol_path, "sbol_multilabels.parquet"))
    sbol_labels = sbol_labels.iloc[:sample]

    sbol_user_features = pd.read_parquet(os.path.join(sbol_path, "user_features.parquet"))

    sbol = sbol_labels.merge(sbol_user_features, on="user_id", how="left")
    sbol = sbol.fillna(0.0)
    sbol["features_part_0"] = sbol[[f'feature_{x}' for x in range(features_count)]].apply(lambda x: list(x), axis=1)

    sbol = sbol[["user_id", "labels", "features_part_0"]]

    users_train, users_test = train_test_split(
        sbol[["user_id"]], shuffle=True, random_state=seed, test_size=0.15
    )

    smm_user_factors = pd.read_parquet(os.path.join(smm_path, "als_user_factors.parquet"))

    sbol = sbol.merge(smm_user_factors, on="user_id", how="left")
    sbol["has_uf"] = ~sbol["user_factors"].isna()
    sbol["has_uf"] = sbol["has_uf"].astype(int)
    sbol["user_factors"] = sbol.apply(lambda x: np.zeros(10) if x["has_uf"] == 0 else x["user_factors"], axis=1)
    sbol["features_part_1"] = sbol.apply(lambda x: np.concatenate((x["user_factors"], np.array([x["has_uf"]])),
                                                                  axis=0), axis=1)

    sbol = sbol[["user_id", "labels", "features_part_0", "features_part_1"]]

    if parts_num == 1:
        sbol["features_part_0"] = sbol.apply(
            lambda x: np.concatenate(
                (x["features_part_0"], x["features_part_1"]), axis=0
            ), axis=1)

    sbol_train = users_train.merge(sbol, on="user_id", how="left")
    sbol_test = users_test.merge(sbol, on="user_id", how="left")

    for part in range(parts_num):

        train_ds = datasets.Dataset.from_dict(
            {"labels": sbol_train["labels"], f"features_part_{part}": sbol_train[f"features_part_{part}"]})
        test_ds = datasets.Dataset.from_dict(
            {"labels": sbol_test["labels"], f"features_part_{part}": sbol_test[f"features_part_{part}"]})

        ds = datasets.DatasetDict({'train_train': train_ds, 'train_val': test_ds})
        ds.save_to_disk(
            os.path.join(data_dir_path, f"vfl_multilabel_sber_sample{sample}_parts{parts_num}", f"part_{part}")
        )


if __name__ == "__main__":
    load_data("/home/dmitriy/data/my_test")
