import os
import warnings
import logging
import gc
import time
from contextlib import contextmanager

import pandas as pd
import datasets
import numpy as np
from sklearn.model_selection import train_test_split

warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


def fillna_func(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    print(f"{df_name}: before fillna: {df.isnull().sum().sum()} NaN")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # convert Inf to Nans
    df_mean_dict = df.mean(axis=0, skipna=True).to_dict()
    df = df.fillna(value=df_mean_dict, method=None, axis=0, inplace=False)
    print(f"{df_name}: after fillna: {df.isnull().sum().sum()} NaN")
    return df


def split_save_datasets(df, train_appid, test_appid, columns, postfix_sample, dir_name_postfix, part_postfix,
                        data_dir_path):
    train_df = train_appid.merge(df, on="SK_ID_CURR", how="left")
    test_df = test_appid.merge(df, on="SK_ID_CURR", how="left")

    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    train_ds = datasets.Dataset.from_dict({col: train_df[col] for col in columns})
    test_ds = datasets.Dataset.from_dict({col: test_df[col] for col in columns})

    ds = datasets.DatasetDict({'train_train': train_ds, 'train_val': test_ds})

    ds.save_to_disk(
        os.path.join(os.path.dirname(data_dir_path), f"home_credit_sample{postfix_sample}_parts{dir_name_postfix}",
                     part_postfix)
    )


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Preprocess application_train.csv and application_test.csv
def application_train_test(data_dir: str, num_rows=None, nan_as_category=False):
    # Read data and merge
    df = pd.read_csv(data_dir + '/application_train.csv', nrows=num_rows)
    test_df = pd.read_csv(data_dir + '/application_test.csv', nrows=num_rows)
    print(f"Train samples: {len(df)}, test samples: {len(test_df)}")
    df = pd.concat([df, test_df], ignore_index=True)

    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']

    # 是否提供往额外的文档资料标识
    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]

    # 个人生活信息的资料标识
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # 按照工作类型计算每类工作的收入中位数
    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)

    # 贷款的信用额度 / 贷款年金
    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']

    # 贷款的信用额度 / 贷款商品价格
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']

    # 文档资料的峰度
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    # 生活资料的求和
    # for live_col in live:
    #     df[live_col] = df[live_col].astype(float)
    #
    # df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)

    # 平均每个孩子平分的收入
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])

    # DAYS_EMPLOYED为申请贷款前开始当前工作的时间
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    # 信用局信用年金 / 总收入
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])

    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())

    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    dropcolum = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4',
                 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
                 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
                 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
                 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
                 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
                 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
    df = df.drop(dropcolum, axis=1)
    del test_df
    gc.collect()
    return df


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(data_dir: str, num_rows=None, nan_as_category=True):
    bureau = pd.read_csv(data_dir + '/bureau.csv', nrows=num_rows)
    bb = pd.read_csv(data_dir + '/bureau_balance.csv', nrows=num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()

    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg


# Preprocess POS_CASH_balance.csv
def pos_cash(data_dir: str, num_rows=None, nan_as_category=True):
    pos = pd.read_csv(data_dir + '/POS_CASH_balance.csv', nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    pos_agg.reset_index(inplace=True)
    return pos_agg


def main(data_dir: str, num_rows: int = None):
    df = application_train_test(data_dir=data_dir, num_rows=num_rows)
    df = df[df['TARGET'].notnull()]
    print("df shape:", df.shape)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(data_dir=data_dir, num_rows=num_rows)
        print("Bureau df shape:", bureau.shape)
        gc.collect()

    with timer("Process POS-CASH balance"):
        pos = pos_cash(data_dir=data_dir, num_rows=num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        gc.collect()

    return df, bureau, pos


def load_data(data_dir_path: str, parts_num: int, sample: int, seed: int, use_bureau: bool = False):
    parent_dir = os.path.dirname(data_dir_path)
    homecredit_data, bureau, pos = main(data_dir=parent_dir,
                                        num_rows=sample if sample != -1 else None)
    if sample == -1:
        sample = homecredit_data.shape[0]
    scores = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']  # add these to other sources and remove from
    other_sources_df = homecredit_data[["SK_ID_CURR", *scores]].copy()
    homecredit_data.drop(scores, axis=1, inplace=True)

    df_labels = homecredit_data[["SK_ID_CURR", "TARGET"]]

    postfix_sample = sample
    if parts_num == 2 and use_bureau:
        postfix_sample = f"{sample}_bureau"
    if parts_num == 2 and not use_bureau:
        postfix_sample = f"{sample}_pos"

    appid_train, appid_test = train_test_split(
        df_labels[["SK_ID_CURR"]], shuffle=True, random_state=seed, test_size=0.15, stratify=df_labels[["TARGET"]]
    )

    if parts_num > 1:
        logger.info("Save vfl dataset labels part...")
        split_save_datasets(df=df_labels, train_appid=appid_train, test_appid=appid_test,
                            columns=["SK_ID_CURR", "TARGET"], postfix_sample=postfix_sample, part_postfix="master_part",
                            dir_name_postfix=parts_num, data_dir_path=data_dir_path)

    # # preparing applications dataframe
    homecredit_data = fillna_func(df=homecredit_data, df_name="Applications")
    cols_to_concat = [c for c in homecredit_data.columns if c not in ["SK_ID_CURR", "TARGET"]]
    homecredit_data["features_part_0"] = homecredit_data[cols_to_concat].apply(
        lambda x: list(x), axis=1)
    homecredit_data = homecredit_data[["SK_ID_CURR", "features_part_0", "TARGET"]]

    if parts_num == 1:
        logger.info("Save applications only dataset for single experiments....")
        split_save_datasets(df=homecredit_data, train_appid=appid_train, test_appid=appid_test,
                            columns=["SK_ID_CURR", "features_part_0", "TARGET"], postfix_sample=postfix_sample,
                            part_postfix="part_0", dir_name_postfix=parts_num, data_dir_path=data_dir_path)

    else:
        logger.info("Save vfl dataset part 0...")
        split_save_datasets(df=homecredit_data, train_appid=appid_train, test_appid=appid_test,
                            columns=["SK_ID_CURR", "features_part_0"], postfix_sample=postfix_sample,
                            part_postfix="part_0", dir_name_postfix=parts_num, data_dir_path=data_dir_path)

    if (parts_num == 2 and use_bureau) or parts_num == 3:

        # preparing bureau dataframe
        bureau = bureau.merge(other_sources_df, on="SK_ID_CURR", how="left")
        bureau = fillna_func(df=bureau, df_name="Bureau")
        cols_to_concat = [c for c in bureau.columns if c not in ["SK_ID_CURR"]]
        bureau["features_part_1"] = bureau[cols_to_concat].apply(
            lambda x: list(x), axis=1)
        bureau = bureau[["SK_ID_CURR", "features_part_1"]]
        logger.info("Save vfl dataset part 1...")
        split_save_datasets(df=bureau,  train_appid=appid_train, test_appid=appid_test,
                            columns=["SK_ID_CURR", "features_part_1"], postfix_sample=postfix_sample,
                            part_postfix="part_1", dir_name_postfix=parts_num, data_dir_path=data_dir_path)

    if (parts_num == 2 and not use_bureau) or parts_num == 3:

        # preparing pos balance dataframe
        pos = fillna_func(df=pos, df_name="POS Balance")
        cols_to_concat = [c for c in pos.columns if c not in ["SK_ID_CURR"]]
        features_part_number = 2 if parts_num == 3 else 1
        pos[f"features_part_{features_part_number}"] = pos[cols_to_concat].apply(
            lambda x: list(x), axis=1)
        pos = pos[["SK_ID_CURR", f"features_part_{features_part_number}"]]

        logger.info(f"Save vfl dataset part {features_part_number}...")
        split_save_datasets(df=pos, train_appid=appid_train, test_appid=appid_test,
                            columns=["SK_ID_CURR", f"features_part_{features_part_number}"],
                            postfix_sample=postfix_sample, part_postfix=f"part_{features_part_number}",
                            dir_name_postfix=parts_num, data_dir_path=data_dir_path)


if __name__ == "__main__":
    load_data(data_dir_path="/home/dmitriy/Projects/vfl-benchmark/experiments/airflow/data/",
              parts_num=3, use_bureau=True, seed=22, sample=2000)

# load_data(data_dir_path="/home/dmitriy/Projects/vfl-benchmark/experiments/airflow/data/home_credit_sample10000_parts_single")
