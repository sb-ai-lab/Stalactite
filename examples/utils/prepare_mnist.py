import os

import numpy as np
import shutil
from pathlib import Path
import argparse

import datasets
from torchvision import transforms
from functools import partial


def substitute(sample, new_labels, use_labels):
    sample['label'] = (new_labels[0] if (sample['label'] == use_labels[0]) else new_labels[1])
    return sample


def make_train_val_split(ds, test_size=0.15, stratify_by_column='label', shuffle=True, seed=47):
    """

    2. Dataset divided into val and train parts.

    """

    d_dict = ds.train_test_split(test_size=test_size, stratify_by_column=stratify_by_column, shuffle=shuffle,
                                 seed=seed)

    train_train = d_dict['train']
    train_train.split._name = 'train_train'

    train_val = d_dict['test']
    train_val.split._name = 'train_val'

    return train_train, train_val


def split_image(image, parts=2):
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
    """

    3. Image divided into different parts.
    First, the image data converted into a tensor.
    Then this data divided into parts in such a way that the dimension of tensor divided by number of parties.
    So each image divided among all parties.
    Image divided vertically: one piece is a vertical strip from image.
    After this, these parts of the tensor are again converted into PIL image.

    """

    to_tensor = transforms.ToTensor()
    to_image = transforms.ToPILImage()

    image = sample[split_feature]
    timage = to_tensor(image)[0, :, :]

    image_parts = {f'{part_prefix}_{ii}': to_image(img) for ii, img in enumerate(split_image(timage, parts=parts))}

    return image_parts


def split_dataset_dict(ds_dict, parts, split_feature='image', part_prefix='image_part'):
    """

    4. DatasetDict formed where keys are the names of parts of the image and the values are object Datasets with data.

    """

    # we assume that datasets in the dict have the same features
    common_features = [ff for ff in list(ds_dict.values())[0].features.keys() if (ff != split_feature)]

    splited_dss = {}
    for val, ds in ds_dict.items():
        ds_splited = ds.map(partial(split_vertically, parts=parts, part_prefix=part_prefix,
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


def save_master_dataset(dataset, path):
    path = Path(path)
    if not path.exists():
        path.mkdir()

    dataset.save_to_disk(path)


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


def load_data(save_path: Path, parts_num: int, binary: bool = True):
    """

    The input is the original MNIST dataset.
    1. Labels filtered and replaced so that the task is binary.

    """
    make_validation = True
    test_size = 0.15
    stratify_by_column = 'label'
    shuffle = True
    seed = 47

    save_dir = Path(f'{save_path}')

    mnist = datasets.load_dataset('mnist')

    if binary:
        # Filter on labels and substitute labels (make the problem binary):
        use_labels = [3, 8]
        new_labels = [-1, 1]
        mnist = mnist.filter(lambda sample: sample['label'] in use_labels, load_from_cache_file=False)
        # Map lables to new labels:
        mnist = mnist.map(partial(substitute, new_labels=new_labels, use_labels=use_labels))

    mnist["train"] = mnist["train"].add_column(name="image_idx", column=[idx for idx in range(len(mnist["train"]))])
    master_dataset = mnist.select_columns(["label"])

    # Split train part into val and train parts:
    # divide onto train and val
    if make_validation:
        train_train, train_val = make_train_val_split(mnist['train'], test_size=test_size,
                                                      stratify_by_column=stratify_by_column, shuffle=shuffle, seed=seed)
        if parts_num != 1:

            train_train_labels = train_train.select_columns(["image_idx", "label"])
            train_val_labels = train_val.select_columns(["image_idx", "label"])

            train_train = train_train.remove_columns("label")
            train_val = train_val.remove_columns("label")
            master_dataset = datasets.DatasetDict({'train_train': train_train_labels, 'train_val': train_val_labels})
            save_master_dataset(master_dataset, path=os.path.join(save_dir, "master_part"))

        mnist = datasets.DatasetDict({'train_train': train_train, 'train_val': train_val})

    # Split the whole dataset:
    rr = split_dataset_dict(mnist, parts=parts_num, split_feature='image', part_prefix='image_part')

    # Save the whole dataset:
    # Saving parameters:
    save_splitted_dataset(rr, path=save_dir, clean_dir=False)
    save_master_dataset(master_dataset, path=os.path.join(save_dir, "master_part"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command line params')

    parser.add_argument('--save_path', type=str, default='~/stalactite_data',
                        help='Path where the splitted data is saved to')
    parser.add_argument('--members_no', type=int, default=3, help='Amount of parties (members)')

    args = parser.parse_args()
    save_path = Path(args.save_path).absolute() / ('mnist_binary38_parts_' + str(args.members_no))
    save_path.mkdir(parents=True, exist_ok=True)

    load_data(save_path, args.members_no)

    print(f"Splitted data is saved to: {save_path}")
