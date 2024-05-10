"""
Utility functions for data manipulation and processing.
"""
import re

import numpy as np
import torch
from torch.utils.data import Dataset

# global variables
_non_utf8_characters_filter = re.compile(pattern=r"[^\x00-\x7F]")
_link_pattern = re.compile(pattern=r"https?://\S+")


# === String ===
def remove_non_utf_8(text):
    """
    Remove the non-UTF-8 characters from the text.

    Args:
        text (str) : The text to remove non-UTF-8 characters.

    Returns:
        str : The text after removing non-UTF-8 characters.
    """
    return _non_utf8_characters_filter.sub(repl="", string=text)


def remove_links(text):
    """
    Remove the links from the text.

    Args:
        text (str) : The text to remove the links.

    Returns:
        str : The text after removing the links.
    """
    return _link_pattern.sub(repl="", string=text)

# === Torch ===
def get_device():
    """
    Get the device for the model.

    Returns:
        torch.device : The device for the model.
    """
    return torch.device(device="cuda" if torch.cuda.is_available() else "cpu")


class ArticleDataset(Dataset):

    def __init__(self, articles, context) -> None:
        """
        Custom dataset to generate the training data for the LLM model.

        Args:
            articles (numpy.ndarray) : The articles to generate the training data.
            context (int) : The context size for the model.
        """
        super().__init__()
        self.x = articles
        self.ctx = context
        self.limit = self.x.shape[1] - self.ctx - 1

    def __len__(self):
        """Get the length of the dataset."""
        return self.x.shape[0]

    def __getitem__(self, index):
        """Get the item from the dataset."""
        if self.limit == 0:
            x = self.x[index, :-1]
            y = self.x[index, 1:]
        else:
            init_index = np.random.randint(low=0, high=self.limit, size=1).item()
            x = self.x[index, init_index:self.ctx]
            y = self.x[index, init_index + 1:self.ctx + 1]

        t_x = torch.tensor(data=x, requires_grad=False).long()
        t_y = torch.tensor(data=y, requires_grad=False).long()

        return t_x, t_y
