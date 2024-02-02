import numpy as np
import shutil
from pathlib import Path

import datasets
from torchvision import transforms
from functools import partial


def substitute(sample, new_labels, use_labels):
    sample['label'] = (new_labels[0] if (sample['label'] == use_labels[0]) else new_labels[1])
    return sample


def make_train_val_split(ds, test_size=0.15, stratify_by_column='label', shuffle=True, seed=47):
    d_dict = ds.train_test_split(test_size=test_size, stratify_by_column=stratify_by_column, shuffle=shuffle,
                                 seed=seed)

    train_train = d_dict['train']
    train_train.split._name = 'train_train'

    train_val = d_dict['test']
    train_val.split._name = 'train_val'

    return train_train, train_val


def split_image(image, parts=2):
    """
    Splits the image onto several parts. Has not been tested when the number of parts larger than 5.
    So, there may be border effects.
    """

    split_dim = 1
    split_dim_size = image.shape[split_dim]

    part_size = int(np.ceil(split_dim_size / parts))
    split_points = np.arange(0, split_dim_size, part_size)

    image_parts = []
    start_points = list(split_points[0:])
    end_points = list(split_points[1:]) + [split_dim_size + 1, ]
    for st, en in zip(start_points, end_points):
        part = image[st:en, :] if (split_dim == 0) else image[:, st:en]
        image_parts.append(part)

    return image_parts


def split_vertically(sample, parts=3, split_feature='image', part_prefix='image_part'):
    to_tensor = transforms.ToTensor()
    to_image = transforms.ToPILImage()

    image = sample[split_feature]
    timage = to_tensor(image)[0, :, :]

    image_parts = {f'{part_prefix}_{ii}': to_image(img) for ii, img in enumerate(split_image(timage, parts=parts))}

    return image_parts


def split_dataset_dict(ds_dict, parts, split_feature='image', part_prefix='image_part'):

    # we assume that datasets in the dict have the same features
    common_features = [ff for ff in list(ds_dict.values())[0].features.keys() if (ff != split_feature)]

    splited_dss = {}
    for val, ds in ds_dict.items():
        ds_splited = ds.map(partial(split_vertically, parts = parts, part_prefix=part_prefix,
                                    split_feature=split_feature), remove_columns=split_feature)
        splited_dss[val] = ds_splited

    splitted_features = [ff for ff in splited_dss[list(splited_dss.keys())[0]].features.keys() if
                         ff not in common_features]

    # seems that only the remove feature we can use here
    parts = []
    for ff in splitted_features:
        features_to_remove = [ee for ee in splitted_features if ee != ff]
        new_ds_dict = datasets.DatasetDict()

        for val, ds in splited_dss.items():
            new_ds_dict[val] = ds.remove_columns(features_to_remove)

        parts.append(new_ds_dict)
    return parts


def save_splitted_dataset(ds_list, path, part_dir_name='part_', clean_dir=False):
    path = Path(path)
    if not path.exists():
        path.mkdir()

    # clean the directory:
    if clean_dir:
        for pt in path.glob('*'):
            if pt.is_file():
                pt.unlink()
            elif pt.is_dir():
                shutil.rmtree(pt)

    for ii, ds in enumerate(ds_list):
        part_dir = f'{part_dir_name}{ii}'
        part_path = path / part_dir
        if part_path.exists():
            raise IOError('Directory already exists')
        part_path.mkdir()
        ds.save_to_disk(part_path)


def load_data(save_path, parts_num):

    # Filter on labels and substitute labels (make the problem binary):
    use_labels = [3, 8]
    new_labels = [-1, 1]

    make_validation = True
    test_size = 0.15
    stratify_by_column = 'label'
    shuffle = True
    seed = 47

    save_dir = Path(f'{save_path}')

    mnist = datasets.load_dataset('mnist')
    mnist = mnist.filter(lambda sample: sample['label'] in use_labels, load_from_cache_file=False)
    # Map lables to new labels:
    mnist = mnist.map(partial(substitute, new_labels = new_labels,use_labels = use_labels))

    # Split train part into val and train parts:
    # divide onto train and val
    if make_validation:
        train_train, train_val = make_train_val_split(mnist['train'], test_size=test_size,
                                                      stratify_by_column=stratify_by_column, shuffle=shuffle, seed=seed)
        mnist = datasets.DatasetDict({'train_train': train_train, 'train_val': train_val, 'test': mnist['test']})

    # Split the whole dataset:
    rr = split_dataset_dict(mnist, parts=parts_num, split_feature='image', part_prefix='image_part')

    # Save the whole dataset:
    # Saving parameters:
    save_splitted_dataset(rr, path=save_dir, clean_dir=False)
