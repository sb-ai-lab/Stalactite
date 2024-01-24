import datasets
import torch

from .base_preprocessor import DataPreprocessor


class SkLearnStandardScaler(DataPreprocessor):
    def __init__(self, input_feature_name=None):
        super().__init__()

        from sklearn.preprocessing import StandardScaler

        self.StandardScaler = StandardScaler

    def fit_transform(self, inp_data):
        self.standard_scaler = self.StandardScaler()

        out_data = self.standard_scaler.fit_transform(inp_data)

        return torch.from_numpy(out_data)

    def transform(self, inp_data):
        out_data = self.standard_scaler.transform(inp_data)

        return torch.from_numpy(out_data)
