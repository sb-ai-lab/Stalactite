import os
import logging

import pandas as pd
import datasets
import numpy as np

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_save_datasets(df, train_users, test_users, columns, postfix_sample, dir_name_postfix, part_postfix,
                        data_dir_path):
    train_df = train_users.merge(df, on="user_id", how="left")
    test_df = test_users.merge(df, on="user_id", how="left")

    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    train_ds = datasets.Dataset.from_dict({col: train_df[col] for col in columns})
    test_ds = datasets.Dataset.from_dict({col: test_df[col] for col in columns})

    ds = datasets.DatasetDict({'train_train': train_ds, 'train_val': test_ds})

    ds.save_to_disk(
        os.path.join(data_dir_path, f"multilabel_sber_sample{postfix_sample}_parts{dir_name_postfix}", part_postfix)
    )


def load_data(data_dir_path: str, parts_num: int = 2, is_single: bool = False, sbol_only: bool = False):
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

    if not is_single:
        logger.info("Save vfl dataset labels part...")
        split_save_datasets(df=sbol_labels, train_users=users_train, test_users=users_test,
                            columns=["user_id", "labels"], postfix_sample=sample, part_postfix="master_part",
                            dir_name_postfix=2, data_dir_path=data_dir_path)

    # preparing sbol user features
    sbol_user_features = pd.read_parquet(os.path.join(sbol_path, "user_features.parquet"))
    sbol_user_features.fillna(0, inplace=True)

    # filtering
    sbol_user_features = sbol_labels[["user_id", "labels"]].merge(sbol_user_features, on="user_id", how="left")

    sbol_user_features["features_part_0"] = sbol_user_features[[f'feature_{x}' for x in range(features_count)]].apply(
        lambda x: list(x), axis=1)
    sbol_user_features = sbol_user_features[["user_id", "features_part_0", "labels"]]

    if not is_single:
        logger.info("Save vfl dataset part 0...")
        split_save_datasets(df=sbol_user_features, train_users=users_train, test_users=users_test,
                            columns=["user_id", "features_part_0"], postfix_sample=sample, part_postfix="part_0",
                            dir_name_postfix=2, data_dir_path=data_dir_path)

        logger.info("Save vfl dataset part 0 for arbiter...")
        split_save_datasets(df=sbol_user_features, train_users=users_train, test_users=users_test,
                            columns=["user_id", "features_part_0", "labels"], postfix_sample=sample,
                            part_postfix="master_part_arbiter", dir_name_postfix=2, data_dir_path=data_dir_path)

    else:
        if sbol_only:
            logger.info("Save sbol only dataset for single experiments....")
            split_save_datasets(df=sbol_user_features, train_users=users_train, test_users=users_test,
                                columns=["user_id", "features_part_0", "labels"], postfix_sample=sample,
                                part_postfix="part_0", dir_name_postfix="_sbol_only", data_dir_path=data_dir_path)

    # preparing smm user features
    smm_user_factors = pd.read_parquet(os.path.join(smm_path, "als_user_factors.parquet"))
    # filtering
    smm_user_factors = sbol_labels[["user_id"]].merge(smm_user_factors, on="user_id", how="inner")
    smm_user_factors.rename(columns={"user_factors": "features_part_1"}, inplace=True)

    if not is_single:
        logger.info("Save vfl dataset part 1...")
        split_save_datasets(df=smm_user_factors, train_users=users_train, test_users=users_test,
                            columns=["user_id", "features_part_1"], postfix_sample=sample, part_postfix="part_1",
                            dir_name_postfix=2, data_dir_path=data_dir_path)

    else:
        if not sbol_only:
            logger.info("Save dataset for single experiments....")
            single_df = sbol_user_features.merge(smm_user_factors, on="user_id", how="left")
            single_df["has_uf"] = ~single_df["features_part_1"].isna()
            single_df["has_uf"] = single_df["has_uf"].astype(int)
            single_df["features_part_1"] = single_df.apply(
                lambda x: np.zeros(10) if x["has_uf"] == 0 else x["features_part_1"], axis=1
            )
            single_df["features_part_1"] = single_df.apply(
                lambda x: np.concatenate((x["features_part_1"], np.array([x["has_uf"]])), axis=0), axis=1)

            single_df["features_part_0"] = single_df.apply(
                lambda x: np.concatenate(
                    (x["features_part_0"], x["features_part_1"]), axis=0
                ), axis=1)

            split_save_datasets(
                df=single_df, train_users=users_train, test_users=users_test,
                columns=["user_id", "features_part_0", "labels"], postfix_sample=sample, dir_name_postfix="_single",
                data_dir_path=data_dir_path, part_postfix="part_0")
