import numpy as np
import torch
import random


def set_seed(seed):
    """
    设置随机数种子，使得结果可以复现
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
