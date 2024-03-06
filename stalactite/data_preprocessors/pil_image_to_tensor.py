from functools import partial

from torchvision import transforms

from .base_preprocessor import DataPreprocessor

to_tensor = transforms.ToTensor()
to_image = transforms.ToPILImage()


class PILImageToTensor(DataPreprocessor):
    def __init__(self, input_feature_name=None, flatten=True):
        super().__init__()

        self.input_feature_name = input_feature_name
        self.flatten = flatten

    def fit_transform(self, inp_data):
        return self._transform(inp_data)

    def transform(self, inp_data):
        return self._transform(inp_data)

    def _transform(self, inp_data):
        res = inp_data.map(
            partial(
                self._convert_to_tensor,
                feature_name=self.input_feature_name,
                feature_name_new=f"{self.input_feature_name}_new",
            ),
            remove_columns=[
                self.input_feature_name,
            ],
        )

        res = res.rename_column(f"{self.input_feature_name}_new", self.input_feature_name)
        res = res.with_format("torch")

        return res

    def _convert_to_tensor(self, sample, feature_name, feature_name_new):

        image = sample[feature_name]  # make sure it is decoded from the storing format
        timage = to_tensor(image)[0, :, :]
        if self.flatten:
            timage = timage.flatten()
        ret = {f"{feature_name_new}": timage}

        return ret
