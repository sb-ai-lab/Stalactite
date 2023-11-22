import torch

from .base_preprocessor import DataPreprocessor


class FullDataTensor(DataPreprocessor):
    def __init__(self, input_feature_name=None):
        super().__init__()

        self.input_feature_name = input_feature_name

    def fit_transform(self, inp_data):
        return self._transform(inp_data)

    def transform(self, inp_data):
        return self._transform(inp_data)

    def _transform(self, inp_data):
        num_rows = inp_data.num_rows

        tnsr = torch.as_tensor(inp_data[self.input_feature_name][0:num_rows])

        return tnsr
