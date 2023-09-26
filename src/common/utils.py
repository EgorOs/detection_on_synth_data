import contextlib
import random
import ssl
from copy import deepcopy

import numpy as np
import torch


@contextlib.contextmanager
def set_unverified_ssl_context():
    orig_context = deepcopy(ssl._create_default_https_context)
    ssl._create_default_https_context = ssl._create_unverified_context
    yield
    ssl._create_default_https_context = orig_context


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
