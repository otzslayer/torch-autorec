from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import yaml


def load_config(config_path: str) -> dict:
    """A simple function to load yaml configuration file

    Parameters
    ----------
    config_path : str
        the path of yaml configuration file
    """
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config


def get_metrics(
    model: nn.Module, train_set=data.Dataset, test_set=data.Dataset
) -> np.float32:

    # Get matrix from torch Dataset
    test_mat = torch.Tensor(test_set.data).cuda()
    test_mask = (test_mat > 0).cuda()

    # Reconstruct the test matrix
    reconstruction = model(test_mat)

    # Get unseen users and items
    unseen_users = test_set.users - train_set.users
    unseen_items = test_set.users - train_set.items

    # Use a default rating of 3 for test users or
    # items without training observations.
    for item, user in product(unseen_items, unseen_users):
        if test_mask[user, item]:
            reconstruction[user, item] = 3

    return masked_rmse(actual=test_mat, pred=reconstruction, mask=test_mask)


def masked_rmse(
    actual: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
) -> np.float32:
    mse = ((pred - actual) * mask).pow(2).sum() / mask.sum()

    return np.sqrt(mse.detach().cpu().numpy())
