import datasets
import torch

from stalactite.data_preprocessors import FullDataTensor, PILImageToTensor, RemoveZeroStdColumns, SkLearnStandardScaler


class ImagePreprocessor:
    def __init__(self, dataset: datasets.DatasetDict,  member_id, data_params=None):
        self.dataset = dataset
        self.data_params = data_params
        self.member_id = member_id

    def fit_transform(self):

        train_split_key = self.data_params.train_split
        test_split_key = self.data_params.test_split

        data_train = self.dataset[train_split_key]
        data_test = self.dataset[test_split_key]

        feature_name = self.data_params.features_key
        label_name = self.data_params.label_key

        image2tensor = PILImageToTensor(feature_name)
        full_data_tensor = FullDataTensor(feature_name)
        remove_zero_std = RemoveZeroStdColumns()
        standard_scaler = SkLearnStandardScaler()

        train_split_data, test_split_data = {}, {}

        for split_dict, split_data in zip([train_split_data, test_split_data], [data_train, data_test]):
            split_dict[feature_name] = image2tensor.fit_transform(split_data)
            for preprocessors in [full_data_tensor, remove_zero_std, standard_scaler]:
                split_dict[feature_name] = preprocessors.fit_transform(split_dict[feature_name])

            split_dict[label_name] = split_data[label_name] #todo: revise#full_data_tensor.fit_transform(torch.tensor(split_data[label_name]))

        ds_train = datasets.Dataset.from_dict(train_split_data, split=train_split_key)
        ds_test = datasets.Dataset.from_dict(test_split_data, split=test_split_key)
        ds = datasets.DatasetDict({train_split_key: ds_train, test_split_key: ds_test})
        ds = ds.with_format("torch")

        return ds


