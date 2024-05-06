import os
os.environ["TOKENIZERS_PARALLELISM"] = "False"

from zipfile import ZipFile
import warnings
import logging
import time
from typing import List
import re
import glob

import pandas as pd
import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import shutil

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logger = logging.getLogger(__name__)

cat_optimal_smoothing = {
    'user_id': 105239,
    'region': 50000,
    'city': 1052390,
    'parent_category_name': 100,
    'category_name': 500,
    'param_1': 1000,
    'param_2': 1000,
    'param_3': 10000,
    'user_type': 100
}


def fillna_scale(df: pd.DataFrame, train_item_id: pd.DataFrame, test_item_id:pd.DataFrame) -> pd.DataFrame:
    df_train = train_item_id.merge(df, on="item_id", how="left")
    df_test = test_item_id.merge(df, on="item_id", how="left")

    df_train.drop("item_id", axis=1, inplace=True)
    df_test.drop("item_id", axis=1, inplace=True)

    df_train_mean_dict = df_train.mean(axis=0, skipna=True).to_dict()
    df_train = df_train.fillna(value=df_train_mean_dict, method=None, axis=0, inplace=False)
    df_test = df_test.fillna(value=df_train_mean_dict, method=None, axis=0, inplace=False)

    # Normalize train and test
    scaler = StandardScaler()
    df_train = pd.DataFrame(scaler.fit_transform(df_train))
    df_test = pd.DataFrame(scaler.transform(df_test))

    df_train["item_id"] = train_item_id["item_id"]
    df_test["item_id"] = test_item_id["item_id"]

    df = pd.concat([df_train, df_test], ignore_index=True)

    return df


def split_save_datasets(df, train_item_id, test_item_id, columns, postfix_sample, dir_name_postfix, part_postfix,
                        data_dir_path):
    train_df = train_item_id.merge(df, on="item_id", how="left")
    test_df = test_item_id.merge(df, on="item_id", how="left")

    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    train_ds = datasets.Dataset.from_dict({col: train_df[col] for col in columns})
    test_ds = datasets.Dataset.from_dict({col: test_df[col] for col in columns})
    ds = datasets.DatasetDict({'train_train': train_ds, 'train_val': test_ds})

    ds.save_to_disk(
        os.path.join(os.path.dirname(data_dir_path), f"avito_sample{postfix_sample}_parts{dir_name_postfix}",
                     part_postfix)
    )


def prepare_cat_features(
        df: pd.DataFrame, train_item_id: pd.DataFrame, test_item_id: pd.DataFrame,
        labels_stratify_df: pd.Series, categorical_features: List[str],
        cat_feature_processed_suffix: str = '_me') -> pd.DataFrame:

    train_df = train_item_id.merge(df, on="item_id", how="left")
    test_df = test_item_id.merge(df, on="item_id", how="left")
    train_labels_stratify_df = train_item_id.merge(labels_stratify_df, on="item_id", how="left")

    y_feature = "deal_probability"
    global_mean = train_df[y_feature].mean()
    skf = StratifiedKFold(n_splits=5)

    # for train
    for cat_feature in categorical_features:
        train_df[cat_feature + cat_feature_processed_suffix] = np.nan
        ind_of_curr_cat_feature = list(train_df.columns).index(cat_feature + cat_feature_processed_suffix)

        smoothing_m = cat_optimal_smoothing[cat_feature]

        for fold_no, (train_index, test_index) in enumerate(
                skf.split(train_df, train_labels_stratify_df["stratify_feature"])
        ):
            skf_train, skf_test = train_df.iloc[train_index], train_df.iloc[test_index]

            # TODO: improve
            train_mean_encoding = (skf_train.loc[:, [cat_feature, y_feature]].groupby(by=cat_feature).mean()[
                                       y_feature] * skf_train.shape[0] + smoothing_m * global_mean) / (
                                              skf_train.shape[0] + smoothing_m)

            new_train_feature = skf_test[cat_feature].map(train_mean_encoding)
            new_train_feature = new_train_feature.fillna(global_mean)

            train_df.iloc[test_index, ind_of_curr_cat_feature] = new_train_feature

    # for test
    for cat_feature in categorical_features:
        smoothing_m = cat_optimal_smoothing[cat_feature]

        train_mean_encoding = (train_df.loc[:, [cat_feature, y_feature]].groupby(
            by=cat_feature).mean()[y_feature] * train_df.shape[0] + smoothing_m * global_mean) / (
                                          train_df.shape[0] + smoothing_m)

        new_test_feature = test_df[cat_feature].map(train_mean_encoding)
        new_test_feature = new_test_feature.fillna(global_mean)
        test_df[cat_feature + cat_feature_processed_suffix] = new_test_feature

    df = pd.concat([train_df, test_df], ignore_index=True)

    return df


def prepare_num_features(df: pd.DataFrame, train_item_id, test_item_id, num_features) -> pd.DataFrame:
    train_df = train_item_id.merge(df, on="item_id", how="left")
    test_df = test_item_id.merge(df, on="item_id", how="left")

    num_quantile_transforms, feature_means = normalize_numerical(
        train_df, num_features, quantile_transformers=None, feature_means=None, transformed_suffix='_qn'
    )
    _, _ = normalize_numerical(
        test_df, num_features, quantile_transformers=None, feature_means=feature_means, transformed_suffix='_qn'
    )

    df = pd.concat([train_df, test_df], ignore_index=True)

    return df


def normalize_numerical(data, num_features,
                        quantile_transformers=None, feature_means=None, transformed_suffix='_qn'):
    if quantile_transformers is None:
        quant_transformers_are_provided = False
        quantile_transformers = []
        feature_means = {}
    else:
        quant_transformers_are_provided = True

    for idx, num_feature in enumerate(num_features):
        if not quant_transformers_are_provided:
            q_trans = QuantileTransformer(output_distribution='normal')
            trans_feature = q_trans.fit_transform(data[num_feature].values.reshape(-1, 1))
            trans_feature_mean = np.nanmean(trans_feature)
            trans_feature = np.nan_to_num(trans_feature, copy=True, nan=trans_feature_mean,
                                          posinf=trans_feature_mean, neginf=trans_feature_mean)
            feature_means[num_feature] = trans_feature_mean
            quantile_transformers.append(q_trans)
        else:
            q_trans = quantile_transformers[idx]
            trans_feature = q_trans.transform(data[num_feature].values.reshape(-1, 1))
            trans_feature_mean = feature_means[num_feature]
            trans_feature = np.nan_to_num(trans_feature, copy=True, nan=trans_feature_mean,
                                          posinf=trans_feature_mean, neginf=trans_feature_mean)

        data[num_feature + transformed_suffix] = trans_feature

    return quantile_transformers, feature_means


def prepare_date_features(data, date_features, processed_suffix='_dd'):
    for date_feature in date_features:
        day = data[date_feature].map(lambda x: x.day)
        week = data[date_feature].map(lambda x: x.week)  # do not use cycle features.
        weekday = data[date_feature].map(lambda x: x.weekday())

        # Day of week features:
        wd_encoder = OneHotEncoder(handle_unknown='error', sparse_output=False)
        wd_columns = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
        dd_1 = pd.DataFrame(wd_encoder.fit_transform(weekday.values.reshape(-1, 1)), index=data.index,
                            columns=[(date_feature + '_' + wd + processed_suffix) for wd in wd_columns])

        # Other features:
        day_cos = np.cos((day - 1) / 30)
        day_sin = np.sin((day - 1) / 30)

        dd_2 = pd.DataFrame(np.array([week.values, day_cos.values, day_sin.values]).T, index=data.index,
                            columns=[(date_feature + '_' + cc + processed_suffix) for cc in
                                     ('week', 'day_cos', 'day_sin')])

        data = pd.concat((data, dd_1, dd_2), axis=1)

        return data


def text_clean(text):
    text = str(text)
    text = text.lower()
    clean1 = re.sub(r"[,.;@#?!&$/]+ *", " ", text)
    clean2 = re.sub(' +', ' ', clean1)
    return clean2


def add_simple_text_features(df):
    df["title"] = df["title"].apply(text_clean)
    df["description"] = df["description"].apply(text_clean)

    df["title_words_length"] = df["title"].apply(lambda x: len(x.split()))
    df["description_words_length"] = df["description"].apply(lambda x: len(x.split()))
    df["title_char_length"] = df["title"].apply(lambda x: len(x))
    df["description_char_length"] = df["description"].apply(lambda x: len(x))
    return df


def add_text_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("encoding description embeddings...")

    start_time = time.time()
    descr_embeddings = model.encode(df["description"].values)
    descr_emb_df = pd.DataFrame(
        descr_embeddings,
        columns=['descr_emb' + str(ii) for ii in range(descr_embeddings.shape[1])])

    print(f"Descripton running time: {time.time() - start_time}")
    print("encoding title embeddings...")
    start_time = time.time()

    title_embeddings = model.encode(df["title"].values)
    title_emb_df = pd.DataFrame(
        title_embeddings,
        columns=['title_emb' + str(ii) for ii in range(title_embeddings.shape[1])])

    print(f"Title running time: {time.time() - start_time}")
    df = pd.concat(
        (
            df.loc[:, ['item_id', 'title', 'description',
                       'title_words_length', 'description_words_length', 'title_char_length',
                       'description_char_length']].reset_index(drop=True),
            descr_emb_df.reset_index(drop=True),
            title_emb_df.reset_index(drop=True)
        ),
        ignore_index=False, axis=1)

    return df


def copy_images_to_separate_folder(df, image_col, orig_folder, new_folder):
    with ZipFile(orig_folder, 'r') as zip:
        all_zip_files = zip.namelist()
        def copy_image(fn):
            if pd.isnull(fn):
                return -1
            else:
                file_name = fn + '.jpg'
                # orig_file = orig_folder + file_name
                # if not os.path.exists(orig_file):
                if file_name not in all_zip_files:
                    return 0
                else:
                    # new_file = new_folder + file_name
                    zip.extract(member=file_name, path=new_folder)
                    # try:
                    #     tt = torch.ops.image.read_file(orig_file)
                    #     shutil.copy2(orig_file, new_folder)
                    # except Exception as e:
                    #     return -2
                    return 1

        if not os.path.exists(new_folder):
            os.makedirs(new_folder, mode=777)  # does not work. Need manually to create a dir with mode=777

        res = df[image_col].map(copy_image)

    return res


class AvitoImagesDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.files = glob.glob(img_dir + '*.jpg')
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx].split('/')[-1][0:-4]
        image = read_image(self.files[idx])
        if self.transform:
            image = self.transform(image)
        return image, file_name


def add_imagenet_probs(df, parent_dir, subsample_image_folder):
    start_time = time.time()

    batch_size = 64
    device = 'cpu'
    predictions_classes_num = 10

    with open(os.path.join(parent_dir, "imagenet_class_index.json")) as f:  # image net classes
        imagenet_labels = eval(f.read())
    imagenet_means = [0.485, 0.456, 0.406]
    imagenet_stds = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(
            mean=imagenet_means,
            std=imagenet_stds)
        ])

    avitoImagesDs = AvitoImagesDataset(subsample_image_folder, transform=transform)
    avito_dataloader = DataLoader(avitoImagesDs, batch_size=batch_size, shuffle=False, num_workers=10)

    model = torchvision.models.resnet50(pretrained=True)
    model.eval()
    predictions = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for step, (inputs, file_names) in enumerate(avito_dataloader):
            inputs = inputs.to(device)

            out = model(inputs)
            _, indices = torch.sort(out, descending=True)
            percentage = torch.nn.functional.softmax(out, dim=1) * 100

            predictions_batch = []
            for (im_ind, sorted_inds) in enumerate(indices):
                tt_1 = [imagenet_labels[str(pred_ind.item())][1] for pred_ind in
                        sorted_inds[0:predictions_classes_num]]
                tt_2 = [percentage[im_ind, pred_ind].item() for pred_ind in sorted_inds[0:predictions_classes_num]]

                file_name = file_names[im_ind]
                predictions_batch.append([file_name, ] + tt_1 + tt_2)

            predictions.extend(predictions_batch)
            if (step % 1000) == 0:
                print(step, time.time() - start_time)
            else:
                print('.', end='')

    column_names = ['image', ] + [f'pred_class_text_{i}' for i in range(predictions_classes_num)] + \
                   [f'pred_class_prob_{i}' for i in range(predictions_classes_num)]

    image_pred_features = pd.DataFrame.from_records(predictions, columns=column_names)

    # Select features
    df = df.merge(
        image_pred_features.loc[:, ['image', 'pred_class_prob_0', 'pred_class_prob_1', 'pred_class_prob_2']],
        on='image', how='left')

    print(f"image running time: {time.time() - start_time}")

    return df


def load_data(data_dir_path: str, parts_num: int = 2, is_single: bool = False, tabular_only: bool = False):
    debug = False
    subsample_size = 0.07
    seed = 22
    sample = 100_000  ##100_000 #10_000 #100_000_000
    parent_dir = os.path.dirname(data_dir_path)
    print("reading dataframe")
    tabular_df = pd.read_csv(os.path.join(parent_dir, "train.csv"), nrows=sample)
    print(tabular_df.shape)
    categorical_features = ['user_id', 'region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2',
                            'param_3', 'user_type']
    numerical_features = ['price', 'item_seq_number']
    date_features = ['activation_date']

    df_labels = tabular_df[["item_id", "deal_probability"]]

    labels_stratify_df = tabular_df[["item_id"]].copy()
    labels_stratify_df["stratify_feature"] = tabular_df["deal_probability"] == 0

    item_id_train, item_id_test = train_test_split(
        tabular_df[["item_id"]], train_size=subsample_size,
        test_size=subsample_size, random_state=seed, stratify=labels_stratify_df["stratify_feature"]
    )
    sample_item_id = pd.concat([item_id_train, item_id_test], ignore_index=True)
    # filtering
    tabular_df = sample_item_id.merge(tabular_df, how="left", on="item_id")
    print("filtering")
    print(tabular_df.shape)
    labels_stratify_df = tabular_df[["item_id"]].copy()
    labels_stratify_df["stratify_feature"] = tabular_df["deal_probability"] == 0

    if not is_single:
        logger.info("Save vfl dataset labels part...")
        split_save_datasets(df=df_labels, train_item_id=item_id_train, test_item_id=item_id_test,
                            columns=["item_id", "deal_probability"], postfix_sample=subsample_size,
                            part_postfix="master_part",
                            dir_name_postfix=3,
                            data_dir_path=data_dir_path)

    # todo: check target distr in train-test split
    # preparing tabular dataframe
    print("preparing tabular dataframe")
    if not debug:
        # categorical features
        tabular_df = prepare_cat_features(
            df=tabular_df, train_item_id=item_id_train, test_item_id=item_id_test,
            labels_stratify_df=labels_stratify_df,
            categorical_features=categorical_features
        )
        # numerical features
        tabular_df = prepare_num_features(
            df=tabular_df, train_item_id=item_id_train, test_item_id=item_id_test, num_features=numerical_features
        )

        # date features
        tabular_df['activation_date'] = pd.to_datetime(tabular_df['activation_date'])
        tabular_df = prepare_date_features(data=tabular_df, date_features=date_features)

        feature_suffixes_to_use = ['_me', '_qn', '_dd']
        features_to_use = [cc for cc in tabular_df.columns if cc[-3:] in feature_suffixes_to_use]
        columns_to_use = ["item_id", *features_to_use, "deal_probability"]
        tabular_df = tabular_df.loc[:, columns_to_use]

        tabular_df["features_part_0"] = tabular_df[features_to_use].apply(
            lambda x: list(x), axis=1)

        tabular_df = tabular_df[["item_id", "features_part_0", "deal_probability"]]

    if not is_single:
        logger.info("Save vfl dataset part 0...")
        split_save_datasets(df=tabular_df, train_item_id=item_id_train, test_item_id=item_id_test,
                            columns=["item_id", "features_part_0"], postfix_sample=subsample_size,
                            part_postfix="part_0", dir_name_postfix=3, data_dir_path=data_dir_path)

    else:
        if tabular_only:
            logger.info("Save sbol only dataset for single experiments....")
            split_save_datasets(df=tabular_df, train_item_id=item_id_train, test_item_id=item_id_test,
                                columns=["item_id", "features_part_0", "deal_probability"], postfix_sample=subsample_size,
                                part_postfix="part_0", dir_name_postfix="_tabular_only",
                                data_dir_path=data_dir_path)

    # preparing text dataframe
    print("preparing text dataframe")
    text_df = pd.read_csv(os.path.join(parent_dir, "train.csv"), nrows=sample)
    text_df = text_df[["item_id", "title", "description"]]
    # filtering
    text_df = sample_item_id.merge(text_df, how="left", on="item_id")
    print("filtering")
    print(text_df.shape)
    if not debug:

        text_df = add_simple_text_features(text_df)
        text_df = add_text_embeddings(text_df)

        features_to_scale = [
            'title_words_length', 'description_words_length', 'title_char_length', 'description_char_length'
        ]
        use_features = ["item_id"] + features_to_scale + \
                       [cc for cc in text_df.columns if cc.startswith('descr_emb')] + [cc for cc in text_df.columns if
                                                                                cc.startswith('title_emb')]
        text_df = text_df.loc[:, use_features]

    # train_Y = train_df['deal_probability']

    # test_X = te_df.loc[:, use_features]
    # test_Y = test_df['deal_probability']

        # Fill missing values:

        text_df_train = item_id_train.merge(text_df, on="item_id", how="left")
        text_df_test = item_id_test.merge(text_df, on="item_id", how="left")
        text_df_train_item_id = text_df_train["item_id"]
        text_df_test_item_id = text_df_test["item_id"]

        text_df_train = text_df_train[[c for c in text_df_train.columns if c != "item_id"]]
        text_df_test = text_df_test[[c for c in text_df_test.columns if c != "item_id"]]

        text_df_mean_dict = text_df_train.mean(axis=0, skipna=True).to_dict()
        text_df_train = text_df_train.fillna(value=text_df_mean_dict, method=None, axis=0, inplace=False)
        text_df_test = text_df_test.fillna(value=text_df_train, method=None, axis=0, inplace=False)

        # Normalize X
        scaler = StandardScaler()
        text_df_train = np.hstack(
            (scaler.fit_transform(text_df_train.loc[:, features_to_scale]),
             text_df_train.loc[:, [cc for cc in text_df_train.columns if not cc in features_to_scale]]))
        text_df_test = np.hstack(
            (scaler.transform(text_df_test.loc[:, features_to_scale]),
             text_df_test.loc[:, [cc for cc in text_df_test.columns if not cc in features_to_scale]]))

        text_df_train = pd.DataFrame(text_df_train)
        text_df_test = pd.DataFrame(text_df_test)

        text_df_train["item_id"] = text_df_train_item_id
        text_df_test["item_id"] = text_df_test_item_id

        text_df = pd.concat([text_df_train, text_df_test], ignore_index=True)

        # # Proper shapes for Y:
        # train_Y = train_Y.to_numpy();
        # test_Y = test_Y.to_numpy();  # Y to numpy
        # train_Y.shape = (train_Y.shape[0], 1);
        # test_Y.shape = (test_Y.shape[0], 1)  # Y to columns

        cols_to_concat = [c for c in text_df.columns if c not in ["item_id"]]
        text_df["features_part_1"] = text_df[cols_to_concat].apply(
            lambda x: list(x), axis=1)
        text_df = text_df[["item_id", "features_part_1"]]

    if not is_single:
        logger.info("Save vfl dataset part 1...")
        split_save_datasets(df=text_df,  train_item_id=item_id_train, test_item_id=item_id_test,
                            columns=["item_id", "features_part_1"], postfix_sample=subsample_size,
                            part_postfix="part_1", dir_name_postfix=3, data_dir_path=data_dir_path)

    # preparing images dataframe
    print("preparing images dataframe")
    images_df = pd.read_csv(os.path.join(parent_dir, "train.csv"), nrows=sample)
    images_df = images_df[["item_id", "image", "image_top_1"]]
    subsample_image_folder = f'train_jpg_seed{seed}_size{subsample_size}/'.replace('.', '_')
    # filtering
    images_df = sample_item_id.merge(images_df, how="left", on="item_id")
    print("filtering")
    print(images_df.shape)

    _ = copy_images_to_separate_folder(
        df=images_df,
        image_col='image',
        orig_folder=os.path.join(parent_dir, 'train_jpg_0.zip'),
        new_folder=os.path.join(parent_dir, subsample_image_folder))

    images_df = add_imagenet_probs(df=images_df, parent_dir=parent_dir,
                                   subsample_image_folder=os.path.join(parent_dir, subsample_image_folder))

    images_df = images_df[['item_id', 'image_top_1', 'pred_class_prob_0', 'pred_class_prob_1', 'pred_class_prob_2']]

    images_df = fillna_scale(df=images_df, train_item_id=item_id_train, test_item_id=item_id_test)

    cols_to_concat = [c for c in images_df.columns if c not in ["item_id"]]
    images_df["features_part_2"] = images_df[cols_to_concat].apply(
        lambda x: list(x), axis=1)
    images_df = images_df[["item_id", "features_part_2"]]

    # # Proper shapes for Y:
    # train_Y = train_Y.to_numpy();
    # test_Y = test_Y.to_numpy();  # Y to numpy
    # train_Y.shape = (train_Y.shape[0], 1);
    # test_Y.shape = (test_Y.shape[0], 1)  # Y to columns

    if not is_single:
        logger.info("Save vfl dataset part 2...")
        split_save_datasets(
            df=images_df, train_item_id=item_id_train, test_item_id=item_id_test,
            columns=["item_id", "features_part_2"], postfix_sample=subsample_size, part_postfix="part_2",
            dir_name_postfix=3, data_dir_path=data_dir_path)

    else:
        if not tabular_only:
            logger.info("Save dataset for single experiments....")
            # join text data
            single_df = tabular_df.merge(text_df, on="item_id", how="left")
            single_df["has_f1"] = ~single_df["features_part_1"].isna()
            single_df["has_f1"] = single_df["has_f1"].astype(int)
            fill_shape = len(text_df["features_part_1"][0])
            single_df["features_part_1"] = single_df.apply(
                lambda x: np.zeros(fill_shape) if x["has_f1"] == 0 else x["features_part_1"],
                axis=1
            )
            single_df["features_part_1"] = single_df.apply(
                lambda x: np.concatenate((x["features_part_1"], np.array([x["has_f1"]])), axis=0), axis=1)

            # join images data
            single_df = single_df.merge(images_df, on="item_id", how="left")
            single_df["has_f2"] = ~single_df["features_part_2"].isna()
            single_df["has_f2"] = single_df["has_f2"].astype(int)
            fill_shape = len(images_df["features_part_2"][0])
            single_df["features_part_2"] = single_df.apply(
                lambda x: np.zeros(fill_shape) if x["has_f2"] == 0 else x["features_part_2"],
                axis=1
            )
            single_df["features_part_2"] = single_df.apply(
                lambda x: np.concatenate((x["features_part_2"], np.array([x["has_f2"]])), axis=0), axis=1)

            single_df["features_part_0"] = single_df.apply(
                lambda x: np.concatenate(
                    (x["features_part_0"], x["features_part_1"], x["features_part_2"]), axis=0
                ), axis=1)

            single_df = single_df[["item_id", "features_part_0", "deal_probability"]]
            a = single_df.isna().sum()

            split_save_datasets(
                df=single_df, train_item_id=item_id_train, test_item_id=item_id_test,
                columns=["item_id", "features_part_0", "deal_probability"],
                postfix_sample=subsample_size, dir_name_postfix="_single",
                data_dir_path=data_dir_path, part_postfix="part_0")


# load_data(
#     data_dir_path="/home/dmitriy/Projects/vfl-benchmark/experiments/airflow/data/avito/avito_sample0_07_parts_single",
#     is_single=True, tabular_only=False
# )
# load_data(
#     data_dir_path="/home/dmitriy/Projects/vfl-benchmark/experiments/airflow/data/avito/avito_sample0_07_parts_tabular_only.csv",
#     is_single=True, tabular_only=True
# )
# load_data(
#     data_dir_path="/home/dmitriy/Projects/vfl-benchmark/experiments/airflow/data/avito/avito_sample0_07_parts3",
#     is_single=False, tabular_only=False
# )