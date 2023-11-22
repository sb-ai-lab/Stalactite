from abc import ABC, abstractmethod


class DataPreprocessor(ABC):
    @abstractmethod
    def fit_transform(self, dataset):
        pass

    @abstractmethod
    def transform(self, dataset):
        pass
