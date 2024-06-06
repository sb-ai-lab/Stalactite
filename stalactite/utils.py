import enum
import math
import os
import random

import numpy as np
import torch
from torch import nn as nn


def init_linear_np(module: nn.Linear, seed: int):
    seed_all(seed)
    init_range = 1.0 / math.sqrt(module.out_features)
    np_uniform = np.random.uniform(low=-init_range, high=init_range, size=module.weight.shape)
    module.weight.data = torch.from_numpy(np_uniform).type(torch.float)


def seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class Role(str, enum.Enum):
    arbiter = "arbiter"
    master = "master"
    member = "member"
