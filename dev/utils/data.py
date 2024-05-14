"""
Utility functions for data manipulation and processing.
"""
import re

import torch
from torch.utils.data import Dataset, random_split

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


class FullDataset(Dataset):

    def __init__(self, articles, **kwargs):
        """
        Custom dataset to generate full data for the LLM model.

        Args:
            articles (numpy.ndarray) : The articles to generate the training data.
        """
        super().__init__()
        self.x = articles

    def __len__(self):
        """Get the length of the dataset."""
        return self.x.shape[0]

    def __getitem__(self, index):
        """Get the item from the dataset."""
        x = self.x[index, :-1]
        y = self.x[index, 1:]

        t_x = torch.tensor(data=x, requires_grad=False).long()
        t_y = torch.tensor(data=y, requires_grad=False).long()

        return t_x, t_y


class ChunkDataset(Dataset):

    def __init__(self, article, **kwargs):
        """
        Custom dataset to generate chunk data for the LLM model.

        Args:
            article (numpy.ndarray) : The article to generate the chunk data.
        """
        super().__init__()
        self.x = article
        self.ctx_len = kwargs.get("context_size")
        assert self.ctx_len < self.x.shape[1], "Context size must be less than the article length."

        self.limit = self.x.shape[1] - self.ctx_len - 1
    
    def __len__(self):
        """Get the length of the dataset."""
        return self.x.shape[0]

    def __getitem__(self, index):
        """Get the item from the dataset."""
        start = torch.randint(low=0, high=self.limit, size=(1,)).item()

        x = self.x[index, start:start + self.ctx_len]
        y = self.x[index, start + 1:start + self.ctx_len + 1]

        t_x = torch.tensor(data=x, requires_grad=False).long()
        t_y = torch.tensor(data=y, requires_grad=False).long()

        return t_x, t_y


def split_data(data, train_size, seed):
    """
    Split the data into training and validation sets.

    Args:
        data (numpy.ndarray) : The data to split.
        train_size (float) : The size of the training set.
        seed (int) : The seed for the random number generator.

    Returns:
        tuple : The training and validation sets.
    """
    # Setting the seed
    generator = torch.Generator().manual_seed(seed)
    
    # Splitting the data
    train_set, valid_set = random_split(dataset=data, lengths=[train_size, 1 - train_size], generator=generator)
    train = data[train_set.indices]
    valid = data[valid_set.indices]

    return train, valid
