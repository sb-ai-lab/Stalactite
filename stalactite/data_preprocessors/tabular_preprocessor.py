import datasets

from stalactite.data_preprocessors import FullDataTensor, RemoveZeroStdColumns, SkLearnStandardScaler


class TabularPreprocessor:
    def __init__(self, dataset: datasets.DatasetDict,  member_id, data_params=None):
        self.dataset = dataset
        self.data_params = data_params.copy()
        self.data_params.features_key = self.data_params.features_key + str(member_id)
        self.member_id = member_id

    def fit_transform(self):

        train_split_key = self.data_params.train_split
        test_split_key = self.data_params.test_split

        data_train = self.dataset[train_split_key]
        data_test = self.dataset[test_split_key]

        feature_name = self.data_params.features_key
        label_name = self.data_params.label_key

        full_data_tensor = FullDataTensor(feature_name)
        standard_scaler = SkLearnStandardScaler()

        train_split_data, test_split_data = {}, {}

        for split_dict, split_data in zip([train_split_data, test_split_data], [data_train, data_test]):
            split_dict[feature_name] = full_data_tensor.fit_transform(split_data)
            split_dict[feature_name] = standard_scaler.fit_transform(split_dict[feature_name])
            split_dict[label_name] = split_data[label_name]

        ds_train = datasets.Dataset.from_dict(train_split_data, split=train_split_key)
        ds_test = datasets.Dataset.from_dict(test_split_data, split=test_split_key)
        ds = datasets.DatasetDict({train_split_key: ds_train, test_split_key: ds_test})
        ds = ds.with_format("torch")

        return ds


