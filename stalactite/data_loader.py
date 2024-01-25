import yaml
import random
import numpy as np
import torch
import datasets
from pathlib import Path
import torchvision.transforms as transforms

from sklearn.preprocessing import StandardScaler


class AttrDict(dict):
    """ Attribute Dictionary"""

    def __setitem__(self, key, value):
        if isinstance(value, dict):  # change dict to AttrDict
            value = AttrDict(value)
        super(AttrDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        value = self.get(key)
        if not isinstance(value, AttrDict):  # this part is need when we initialized the AttrDict datastructure form recursice dict.
            if isinstance(value, dict):  # dinamically change the type of value from dict to AttrDict
                value = AttrDict(value)
                self.__setitem__(key, value)
        return self.get(key)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __getattr__(self, key):
        return self.__getitem__(key)

    __delattr__ = dict.__delitem__

    def __add__(self, other):
        return AttrDict({**self, **other})

def load_yaml_config(yaml_path):
    with open(yaml_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError("Yaml error - check yaml file")

def global_seed(config_path):
    seed = config_path.common.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def init():
    config = AttrDict(load_yaml_config("../experiments/configs/config_local_mnist.yaml"))
    global_seed(config)

    joint_config = AttrDict({})
    for ii in range(config.common.parties_num):
        joint_config[ii] = config
        joint_config[ii].data.features_key += str(ii)
        joint_config[ii].data.dataset = f"mnist_binary38_parts{config.common.parties_num}"

    joint_config['joint_config'] = True
    joint_config['parties'] = list(range(config.common.parties_num))
    return joint_config


def load(args):
    data = {}
    for ii in args['parties']:
        part_path =  Path(args[ii].data.data_dir) / args[ii].data.dataset / f'{args[ii].data.dataset_part_prefix}{ii}'
        data[ii] = datasets.load_from_disk(part_path)
    return data

class DataPreprocessor:
    def __init__(self, dataset, data_params, member_id):
        self.dataset = dataset
        self.data_params = data_params
        self.preprocessors_params = {}
        self.data_preprocessed = False
        self.member_id = member_id

    def preprocess_simple(self):

        train_split_key = self.data_params.train_split # = "train_train" (from config)
        test_split_key = self.data_params.test_split # = "train_val" (from_config)

        data_train = self.dataset[self.member_id][train_split_key]
        data_test = self.dataset[self.member_id][test_split_key]

        feature_name = self.data_params.features_key # = "image_part_" (from config)
        label_name = self.data_params.label_key # = "label" (from_config)

        sc_X = StandardScaler()
        to_tensor = transforms.ToTensor()

        train_split_data = {}

        #PILImageToTensor(input_feature_name = feature_name)
        def image_func(data):
            timage = to_tensor(data[feature_name])[0, :, :].flatten()
            res_dic = {f'{feature_name}': timage}
            return res_dic

        res = data_train.map(image_func)
        train_split_data[feature_name] = res.with_format("torch")

        #FullDataTensor(input_feature_name=feature_name)
        def FullDataTensor_func(data, name):
            num_rows = data.num_rows
            data = torch.as_tensor(data[name][0:num_rows])
            return(data)

        train_split_data[feature_name] = FullDataTensor_func(train_split_data[feature_name], feature_name)

        #RemoveZeroStdColumns()
        nonzero_std_columns = (train_split_data[feature_name].std(axis=0) != 0)
        train_split_data[feature_name] = train_split_data[feature_name][:, nonzero_std_columns]

        #SkLearnStandardScaler
        train_split_data[feature_name] = sc_X.fit_transform(train_split_data[feature_name])

        #FullDataTensor(input_feature_name=label_name)
        train_split_data[label_name] = FullDataTensor_func(data_train, label_name)



        test_split_data = {}

        # PILImageToTensor(input_feature_name = feature_name)
        res = data_test.map(image_func, remove_columns=[feature_name])
        test_split_data[feature_name] = res.with_format("torch")

        # FullDataTensor(input_feature_name=feature_name)
        test_split_data[feature_name] = FullDataTensor_func(test_split_data[feature_name], feature_name)

        # RemoveZeroStdColumns()
        test_split_data[feature_name] = test_split_data[feature_name][:, nonzero_std_columns]

        # SkLearnStandardScaler
        test_split_data[feature_name] = sc_X.transform(test_split_data[feature_name])

        # FullDataTensor(input_feature_name=label_name)
        test_split_data[label_name] = FullDataTensor_func(data_test, label_name)

        ds_train = datasets.Dataset.from_dict(train_split_data, split=train_split_key)
        ds_test = datasets.Dataset.from_dict(test_split_data, split=test_split_key)
        ds = datasets.DatasetDict({train_split_key: ds_train, test_split_key: ds_test})
        ds = ds.with_format("torch")

        self.data_preprocessed = True
        return ds
