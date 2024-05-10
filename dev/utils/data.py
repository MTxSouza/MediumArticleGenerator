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

    def __init__(self, articles, context, pad_index) -> None:
        """
        Custom dataset to generate the training data for the LLM model.

        Args:
            articles (numpy.ndarray) : The articles to generate the training data.
            context (int) : The context size for the model.
            pad_index (int) : The index for padding.
        """
        super().__init__()
        self.x = articles
        self.ctx = context
        self.limit = self.x.shape[1]
        self.pad_index = pad_index

    def __len__(self):
        """Get the length of the dataset."""
        return self.x.shape[0]

    def __getitem__(self, index):
        """Get the item from the dataset."""
        # Get number of padding indices
        data = self.x[index, :]
        pad_indices, = np.where(data == self.pad_index)
        if pad_indices.size:
            first_pad_index = pad_indices[0].item()
        else:
            first_pad_index = 0

        # Compute the limit for the random index
        if first_pad_index > self.ctx:
            limit = first_pad_index - self.ctx
        else:
            limit = 1
        print(limit, first_pad_index, self.ctx)
        init_index = np.random.randint(low=0, high=limit, size=1).item()

        x = data[init_index:init_index + self.ctx]
        y = data[init_index + 1:init_index + self.ctx + 1]

        t_x = torch.tensor(data=x, requires_grad=False).long()
        t_y = torch.tensor(data=y, requires_grad=False).long()

        return t_x, t_y
