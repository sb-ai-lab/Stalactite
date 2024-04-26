import os
import logging

import pandas as pd
import datasets
import numpy as np
from datasets import DatasetDict

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
        os.path.join(os.path.dirname(data_dir_path), f"multilabel_sber_sample{postfix_sample}_parts{dir_name_postfix}",
                     part_postfix)
    )


def load_data(data_dir_path: str, parts_num: int, sample: int, seed: int, use_smm: bool = False):
    sbol_path = os.path.join(os.path.dirname(data_dir_path), "sbol")
    smm_path = os.path.join(os.path.dirname(data_dir_path), "smm")
    zvuk_path = os.path.join(os.path.dirname(data_dir_path), "zvuk")

    features_count = 1345
    # sample = 5_000  #10_000 # -1
    # seed = 22

    # preparing labels
    sbol_labels = pd.read_parquet(os.path.join(sbol_path, "sbol_multilabels.parquet"))
    if sample == -1:
        sample = sbol_labels.shape[0]  #190439

    postfix_sample = sample
    if parts_num == 2 and use_smm:
        postfix_sample = f"{sample}_smm"
    if parts_num == 2 and not use_smm:
        postfix_sample = f"{sample}_zvuk"

    sbol_labels = sbol_labels.iloc[:sample]

    users_train, users_test = train_test_split(
        sbol_labels[["user_id"]], shuffle=True, random_state=seed, test_size=0.15
    )

    if parts_num > 1:
        logger.info("Save vfl dataset labels part...")
        split_save_datasets(df=sbol_labels, train_users=users_train, test_users=users_test,
                            columns=["user_id", "labels"], postfix_sample=postfix_sample, part_postfix="master_part",
                            dir_name_postfix=parts_num, data_dir_path=data_dir_path)

    # preparing sbol user features
    sbol_user_features = pd.read_parquet(os.path.join(sbol_path, "user_features.parquet"))
    sbol_user_features.fillna(0, inplace=True)

    # filtering
    sbol_user_features = sbol_labels[["user_id", "labels"]].merge(sbol_user_features, on="user_id", how="left")

    sbol_user_features["features_part_0"] = sbol_user_features[[f'feature_{x}' for x in range(features_count)]].apply(
        lambda x: list(x), axis=1)
    sbol_user_features = sbol_user_features[["user_id", "features_part_0", "labels"]]

    if parts_num == 1:
        logger.info("Save sbol only dataset for single experiments....")
        split_save_datasets(df=sbol_user_features, train_users=users_train, test_users=users_test,
                            columns=["user_id", "features_part_0", "labels"], postfix_sample=postfix_sample,
                            part_postfix="part_0", dir_name_postfix=parts_num, data_dir_path=data_dir_path)
    else:
        logger.info("Save vfl dataset part 0...")
        split_save_datasets(df=sbol_user_features, train_users=users_train, test_users=users_test,
                            columns=["user_id", "features_part_0"], postfix_sample=postfix_sample, part_postfix="part_0",
                            dir_name_postfix=parts_num, data_dir_path=data_dir_path)

    if (parts_num == 2 and use_smm) or parts_num == 3:
        # preparing smm user features
        smm_user_factors = pd.read_parquet(os.path.join(smm_path, "als_user_factors.parquet"))
        # filtering
        smm_user_factors = sbol_labels[["user_id"]].merge(smm_user_factors, on="user_id", how="inner")
        smm_user_factors.rename(columns={"user_factors": "features_part_1"}, inplace=True)
        logger.info("Save vfl dataset part 1...")
        split_save_datasets(df=smm_user_factors, train_users=users_train, test_users=users_test,
                            columns=["user_id", "features_part_1"], postfix_sample=postfix_sample,
                            part_postfix="part_1",
                            dir_name_postfix=parts_num, data_dir_path=data_dir_path)

    if (parts_num == 2 and not use_smm) or parts_num == 3:
        # preparing zvuk user features
        zvuk_user_factors = pd.read_parquet(os.path.join(zvuk_path, "als_user_factors_zvuk.parquet"))
        # filtering
        zvuk_user_factors = sbol_labels[["user_id"]].merge(zvuk_user_factors, on="user_id", how="inner")
        features_part_number = 2 if parts_num == 3 else 1

        zvuk_user_factors.rename(columns={"user_factors": f"features_part_{features_part_number}"}, inplace=True)

        logger.info(f"Save vfl dataset part {features_part_number}...")
        split_save_datasets(df=zvuk_user_factors, train_users=users_train, test_users=users_test,
                            columns=["user_id", f"features_part_{features_part_number}"], postfix_sample=postfix_sample,
                            part_postfix=f"part_{features_part_number}",
                            dir_name_postfix=parts_num, data_dir_path=data_dir_path)


if __name__ == "__main__":
    load_data(data_dir_path="/home/dmitriy/Projects/vfl-benchmark/experiments/airflow/data/",
              parts_num=1, use_smm=True, seed=22, sample=5000)
    # ds = datasets.load_from_disk(
    #     os.path.join(
    #         "/home/dmitriy/Projects/vfl-benchmark/experiments/airflow/data/multilabel_sber_sample5000_parts3/part_2"
    #     )
    # )
    # print(ds["train_val"])
    # print(len(ds["train_val"]["features_part_2"][0]))
    # print(ds["train_val"]["features_part_2"][0][:10])
