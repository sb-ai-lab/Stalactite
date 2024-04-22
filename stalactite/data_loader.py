from typing import Dict

import datasets
import numpy as np

from stalactite import data_preprocessors


def compute_dataset_info_params(ds, features_prefix, label_prefix=None):
    def calc_dim(sample):
        def calc_dim_multidim(sample, size_name):
            if hasattr(sample, size_name):
                size = getattr(sample, size_name)
                try:
                    size = size()
                except Exception:
                    pass
                if len(size) == 0:
                    return 1
                else:
                    return int(np.prod(size))
            return None

        size_names = ("size", "shape")
        for nn in size_names:
            res = calc_dim_multidim(sample, nn)
            if not res is None:
                return res

        sample = np.array(sample)
        res = calc_dim_multidim(sample, "shape")

        return res


class DataPreprocessor:
    config_features_preprocessor_key = "features_data_preprocessors"  # config key
    config_label_preprocessor_key = "label_data_preprocessors"  # config key

    def __init__(self, dataset, data_params, member_id):
        self.dataset = dataset
        self.data_params = data_params
        self.preprocessors_params = {}
        self.data_preprocessed = False
        self.member_id = member_id

    def preprocess(self):
        """
        Preprocesses train and possibly test data.

        Args:
            preprocess_test (bool, optional): Whether test data is preprocessed. Defaults to False.
        """

        # import pdb; pdb.set_trace()

        train_split_key = self.data_params.train_split  # config key
        test_split_key = self.data_params.test_split  # config key

        self._search_and_fill_preprocessor_params()
        if len(self.preprocessors_params) != 0:
            train_split_data, preprocessors_dict = self._preprocess_split(
                self.dataset[self.member_id][train_split_key], self.preprocessors_params
            )
            test_split_data, _ = self._preprocess_split(
                self.dataset[self.member_id][test_split_key],
                self.preprocessors_params,
                preprocessors_dict=preprocessors_dict,
            )

            ds_train = datasets.Dataset.from_dict(train_split_data, split=train_split_key)
            ds_test = datasets.Dataset.from_dict(test_split_data, split=test_split_key)
            ds = datasets.DatasetDict({train_split_key: ds_train, test_split_key: ds_test})
            ds = ds.with_format("torch")

            # import pdb; pdb.set_trace()
            updated_dataset_params = compute_dataset_info_params(
                ds, self.data_params.features_prefix, self.data_params.label_prefix
            )

            self.trained_preprocessors = preprocessors_dict

            self.data_preprocessed = True
            return ds, updated_dataset_params
        else:
            self.data_preprocessed = True
            return self.dataset, self.data_params

    def _search_and_fill_preprocessor_params(self):

        if self.data_params[DataPreprocessor.config_features_preprocessor_key] is not None:
            self.preprocessors_params[self.data_params.features_key] = self.data_params[
                DataPreprocessor.config_features_preprocessor_key
            ]
        if self.data_params.label_key is not None:
            if self.data_params[DataPreprocessor.config_label_preprocessor_key] is not None:
                self.preprocessors_params[self.data_params.label_key] = self.data_params[
                    DataPreprocessor.config_label_preprocessor_key
                ]

    def _preprocess_split(
            self, dataset_split: datasets.Dataset, preprocessors_params: Dict, preprocessors_dict: Dict = None
    ):

        preprocessed_split_data = {}
        if preprocessors_dict is None:
            preprocessors_dict = {}

        # import pdb; pdb.set_trace()
        for key, pp_params in preprocessors_params.items():
            trained_preprocessors = preprocessors_dict.get(key, None)
            feature_data, preprocessors = self._preprocess_feature(
                dataset_split, key, pp_params, self.data_params, trained_preprocessors=trained_preprocessors
            )
            preprocessed_split_data[key] = feature_data
            if (trained_preprocessors is None) or (len(trained_preprocessors) < 1):
                preprocessors_dict[key] = preprocessors

        return preprocessed_split_data, preprocessors_dict

    @staticmethod
    def _preprocess_feature(
            ds_split: datasets.Dataset,
            feature_name: str,
            preprocessors_params: Dict,
            data_params: Dict,
            trained_preprocessors: bool = None,
    ):
        # import pdb; pdb.set_trace()
        out_data = ds_split
        if (trained_preprocessors is None) or (len(trained_preprocessors) < 1):
            trained_preprocessors = []
            for pp in preprocessors_params:
                preprocessor_class, preprocessor_input = DataPreprocessor.parse_preprocessor_string(pp)
                if preprocessor_input is not None:
                    preprocessor_input = getattr(data_params, preprocessor_input)
                    assert (
                            preprocessor_input == feature_name
                    ), f"Inferred '{preprocessor_input}' and passed '{feature_name}' feature names must be equal!"
                preprocessor_class = getattr(data_preprocessors, preprocessor_class)

                preprocessor_obj = (
                    preprocessor_class(preprocessor_input) if (preprocessor_input is not None) else preprocessor_class()
                )

                out_data = preprocessor_obj.fit_transform(out_data)
                trained_preprocessors.append(preprocessor_obj)

        else:
            for pp in trained_preprocessors:
                out_data = pp.transform(out_data)

        return out_data, trained_preprocessors

    @staticmethod
    def parse_preprocessor_string(preprocessor_string: str):

        open_bracket_position = preprocessor_string.find("(")
        if open_bracket_position != -1:
            close_bracket_position = preprocessor_string.find(")")
            if close_bracket_position != -1:
                preprocessor_class = preprocessor_string[0:open_bracket_position]
                input_parameter = preprocessor_string[open_bracket_position + 1: close_bracket_position]
            else:
                raise ValueError(f"Data Preprocessor {preprocessor_string} inconsistent. No closing bracket!")
        else:
            preprocessor_class = preprocessor_string
            input_parameter = None
        return preprocessor_class, input_parameter
