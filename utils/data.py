from typing import List, Tuple

import numpy as np
import pandas as pd
import torch.utils.data as data
from sklearn.model_selection import train_test_split


class AutoRecData(data.Dataset):
    r"""_summary_

    Parameters
    ----------
    data : np.ndarray
        A two-dimensional list is required.
    """

    def __init__(self, data: np.ndarray) -> None:
        super(AutoRecData, self).__init__()

        self.data = data
        self.items = set(data.nonzero()[0])
        self.users = set(data.nonzero()[1])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> None:
        return self.data[index]


def preprocess_data(
    data: pd.DataFrame, test_size: float = 0.1, random_state: int = 0
) -> Tuple[List[List[int]], List[List[int]], int, int]:
    r"""Helper function to preprocess data.

    Parameters
    ----------
    data : pd.DataFrame
        ML-1M data which consists of user_id, item_id, rating.

    test_size : float, optional
        the proportion of the dataset to include in the test split,
        by default 0.1

    random_state : int, optional
        Seed for shuffling data, by default 0

    Returns
    -------
    Tuple[List[List[int]], List[List[int]], int, int]
        _description_
    """
    num_users = np.max(data["user"])
    num_items = np.max(data["item"])

    # Every user and item would be used as index
    data["user"] -= 1
    data["item"] -= 1

    train, test = train_test_split(
        data.values, test_size=test_size, random_state=random_state
    )

    train_mat = np.zeros((num_users, num_items))
    test_mat = np.zeros((num_users, num_items))

    for user, item, rating in train:
        train_mat[user, item] = rating
    for user, item, rating in test:
        test_mat[user, item] = rating

    return train_mat, test_mat, num_users, num_items
