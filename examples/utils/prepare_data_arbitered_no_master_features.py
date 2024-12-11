import os
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from .prepare_sbol import split_save_datasets

logger = logging.getLogger(__name__)


def load_data(data_dir_path: str, sample: int = -1, seed: int = 0, ):
    sbol_path = os.path.join(os.path.dirname(data_dir_path), "sbol")
    features_count = 1345
    sbol_labels = pd.read_parquet(os.path.join(sbol_path, "sbol_multilabels.parquet"))
    if sample == -1:
        sample = sbol_labels.shape[0]  # 190439

    sbol_labels = sbol_labels.iloc[:sample]

    users_train, users_test = train_test_split(
        sbol_labels[["user_id"]], shuffle=True, random_state=seed, test_size=0.15
    )

    # preparing sbol user features
    sbol_user_features = pd.read_parquet(os.path.join(sbol_path, "user_features.parquet"))
    sbol_user_features.fillna(0, inplace=True)

    # filtering
    sbol_user_features = sbol_labels[["user_id", "labels"]].merge(sbol_user_features, on="user_id", how="left")

    sbol_user_features["features_part_1"] = sbol_user_features[[f'feature_{x}' for x in range(features_count)]].apply(
        lambda x: list(x), axis=1)
    sbol_user_features["features_part_0"] = [[0.] for _ in range(sbol_user_features.shape[0])]

    sbol_user_features = sbol_user_features[["user_id", "features_part_0", "features_part_1", "labels"]]
    print("Save vfl dataset part 0...", data_dir_path, sbol_user_features.shape, sbol_user_features.columns)
    logger.info("Save vfl dataset part 0...")
    split_save_datasets(df=sbol_user_features, train_users=users_train, test_users=users_test,
                        columns=["user_id", "features_part_1"], postfix_sample="master_no_labels",
                        part_postfix="part_1", dir_name_postfix=2, data_dir_path=data_dir_path)

    logger.info("Save vfl dataset for master...")
    print("Save vfl dataset for master...", data_dir_path, sbol_user_features.shape, sbol_user_features.columns)
    split_save_datasets(df=sbol_user_features, train_users=users_train, test_users=users_test,
                        columns=["user_id", "features_part_0", "labels"],
                        postfix_sample="master_no_labels",
                        part_postfix="master_part_arbiter",
                        dir_name_postfix=2,
                        data_dir_path=data_dir_path)
