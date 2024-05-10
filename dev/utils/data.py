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
    return torch.device(name="cuda" if torch.cuda.is_available() else "cpu")


class ArticleDataset(Dataset):

    def __init__(self, articles, context, n_iter, batch_size) -> None:
        """
        Custom dataset to generate the training data for the LLM model.

        Args:
            articles (list[int]) : The articles to generate the training data.
            context (int) : The context size for the model.
            n_iter (int) : Number of iterations in this set.
            batch_size (int) : The batch size for the training data.
        """
        super().__init__()
        self.len = int(n_iter * batch_size)
        self.x = articles
        self.ctx = context
        self.limit = len(articles) - self.ctx - 1

    def __len__(self):
        """Get the length of the dataset."""
        return self.len

    def __getitem__(self, index):
        """Get the item from the dataset."""
        index_content = np.random.randint(low=0, high=self.limit, size=1).item()

        x = self.x[index_content:index_content+self.ctx]
        y = self.x[index_content+1:index_content+self.ctx+1]

        np_x = np.asarray(a=x, dtype=np.int64)
        np_y = np.asarray(a=y, dtype=np.int64)

        t_x = torch.tensor(data=np_x, requires_grad=False).long()
        t_y = torch.tensor(data=np_y, requires_grad=False).long()

        return t_x, t_y
