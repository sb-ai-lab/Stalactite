import torch
import numpy as np
import datasets

from stalactite.data_preprocessors import FullDataTensor, SkLearnStandardScaler


class TabularPreprocessor:
    def __init__(self, dataset: datasets.DatasetDict,  member_id, params=None, is_master: bool = False):
        self.dataset = dataset
        self.common_params = params.vfl_model
        self.data_params = params.data.copy()
        self.data_params.features_key = self.data_params.features_key + str(member_id)
        self.member_id = member_id
        self.multilabel = False
        self.is_master = is_master
    def fit_transform(self):

        train_split_key = self.data_params.train_split
        test_split_key = self.data_params.test_split

        data_train = self.dataset[train_split_key]
        data_test = self.dataset[test_split_key]

        feature_name = self.data_params.features_key
        label_name = self.data_params.label_key
        uids_name = self.data_params.uids_key

        full_data_tensor = FullDataTensor(feature_name)
        standard_scaler = SkLearnStandardScaler()

        train_split_data, test_split_data = {}, {}

        for split_dict, split_data in zip([train_split_data, test_split_data], [data_train, data_test]):
            if self.is_master:
                split_dict[label_name] = split_data[label_name]

            else:
                split_dict[feature_name] = full_data_tensor.fit_transform(split_data)
                split_dict[feature_name] = standard_scaler.fit_transform(split_dict[feature_name])

            split_dict[uids_name] = split_data[uids_name]
            # split_dict[label_name] = [x[6] for x in split_dict[label_name]]  # todo: remove (for debugging only)

        if self.is_master and isinstance(train_split_data[label_name][0], list):
            self.multilabel = True
        ds_train = datasets.Dataset.from_dict(train_split_data, split=train_split_key)
        ds_test = datasets.Dataset.from_dict(test_split_data, split=test_split_key)
        ds = datasets.DatasetDict({train_split_key: ds_train, test_split_key: ds_test})
        ds = ds.with_format("torch")
        self._ds = ds
        return ds

    def get_class_weights(self):

        pos_weights_list = []
        y_train = self._ds[self.data_params.train_split][self.data_params.label_key]
        if self.multilabel:
            classes_idx = [x for x in range(self._ds[self.data_params.train_split][self.data_params.label_key].shape[1])]

            for i, c_idx in enumerate(classes_idx):
                unique, counts = np.unique(y_train[:, i], return_counts=True)
                if unique.shape[0] < 2:
                    pos_weights_list.append(max(pos_weights_list))
                    continue
                pos_weights_list.append(counts[0] / counts[1])
        else:
            unique, counts = np.unique(y_train, return_counts=True)
            pos_weights_list.append(counts[0] / counts[1])

        return torch.tensor(pos_weights_list)


