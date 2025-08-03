import torch
import random
import numpy as np
from config.settings import Config

def set_seeds():
    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    random.seed(Config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.random_seed)