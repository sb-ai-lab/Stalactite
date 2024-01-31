import datasets
import torch
import numpy as np

from stalactite.data_preprocessors import FullDataTensor, PILImageToTensor, SkLearnStandardScaler


class ImagePreprocessor:
    def __init__(self, dataset: datasets.DatasetDict,  member_id, params=None):
        self.dataset = dataset
        self.common_params = params.common
        self.data_params = params.data.copy()
        self.data_params.features_key = self.data_params.features_key + str(member_id)
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
        standard_scaler = SkLearnStandardScaler()

        train_split_data, test_split_data = {}, {}

        for split_dict, split_data in zip([train_split_data, test_split_data], [data_train, data_test]):
            split_dict[feature_name] = image2tensor.fit_transform(split_data)
            for preprocessors in [full_data_tensor, standard_scaler]:
                split_dict[feature_name] = preprocessors.fit_transform(split_dict[feature_name])

            split_dict[label_name] = split_data[label_name]

        ds_train = datasets.Dataset.from_dict(train_split_data, split=train_split_key)
        ds_test = datasets.Dataset.from_dict(test_split_data, split=test_split_key)
        ds = datasets.DatasetDict({train_split_key: ds_train, test_split_key: ds_test})
        ds = ds.with_format("torch")
        self._ds = ds
        return ds

    def get_class_weights(self):

        y_train = self._ds[self.data_params.train_split][self.data_params.label_key]
        pos_weights_list = []
        unique, counts = np.unique(y_train, return_counts=True)
        pos_weights_list.append(counts[0] / counts[1])
        return torch.tensor(pos_weights_list)
