import datasets
import torch
import numpy as np

from stalactite.data_preprocessors import FullDataTensor, PILImageToTensor, SkLearnStandardScaler


class ImagePreprocessorEff:
    """
    Make data preprocessing on data after downloading.
    """
    def __init__(self, dataset: datasets.DatasetDict,  member_id, params=None, is_master: bool = False,
                 master_has_features: bool = False):
        self.dataset = dataset
        self.common_params = params.vfl_model
        self.data_params = params.data.copy()
        self.data_params.features_key = self.data_params.features_key + str(member_id)
        self.member_id = member_id
        self.is_master = is_master
        self.master_has_features = master_has_features

    def fit_transform(self):
        """

        The input is the training and test data in the form of a data set for each participant with pictures separated vertically.
        Using PILImageToTensor, one PIL image from the dataset is converted into a tensor, and we get object Dataset.
        Next, using FullDataTensor, the entire object Dataset is converted into one tensor.
        Afterward, only those columns are saved in the tensor whose average deviation along the 0 axis is not equal to zero.
        Then is the operation SkLearnStandardScaler (changing the size of the distribution of values occurs so that the average value of the observed values is equal to 0)

        """
        train_split_key = self.data_params.train_split
        test_split_key = self.data_params.test_split

        data_train = self.dataset[train_split_key].select([x for x in range(10_000)])
        data_test = self.dataset[test_split_key]

        feature_name = self.data_params.features_key
        label_name = self.data_params.label_key
        uids_name = self.data_params.uids_key

        image2tensor = PILImageToTensor(feature_name, flatten=False)

        train_split_data, test_split_data = {}, {}

        for split_dict, split_data in zip([train_split_data, test_split_data], [data_train, data_test]):

            if self.is_master:
                split_dict[label_name] = split_data[label_name]

            if self.master_has_features or not self.is_master:
                split_dict[feature_name] = image2tensor.fit_transform(split_data)[feature_name]
                features = split_dict[feature_name]
                features = torch.reshape(features, (features.shape[0], 1, features.shape[1], features.shape[2]))
                split_dict[feature_name] = features

            split_dict[uids_name] = split_data[uids_name]

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
        counts_max = max(counts)
        for class_count in counts:
            pos_weights_list.append(counts_max / class_count)
        return torch.tensor(pos_weights_list)
