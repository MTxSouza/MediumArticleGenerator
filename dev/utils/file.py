"""
Utility functions for file operations.
"""
import csv
import json
import os

import numpy as np


# === Directory ===
def create_directory(directory):
    """
    Create the directory if it doesn't exists.

    Args:
        directory (str) : The path to the directory.
    """
    os.makedirs(name=directory, exist_ok=True)

# === Numpy ===
def save_numpy_file(filepath, data):
    """
    Save the data as compressed numpy file.

    Args:
        filepath (str) : The path to the compressed numpy file.
        data (np.ndarray) : The data to save as compressed numpy file.
    """
    np.savez_compressed(file=filepath, data=data)

def load_numpy_file(filepath):
    """
    Load the compressed numpy file.

    Args:
        filepath (str) : The path to the compressed numpy file.

    Returns:
        np.ndarray : The data from the compressed numpy file.
    """
    return np.load(file=filepath)["data"]

# === JSON ===
def save_json_file(filepath, data):
    """
    Save the data as JSON file.

    Args:
        filepath (str) : The path to the JSON file.
        data (dict) : The data to save as JSON file.
    """
    with open(file=filepath, mode="w", encoding="utf-8") as file_buffer:
        json.dump(obj=data, fp=file_buffer, indent=4)

def load_json_file(filepath):
    """
    Load the JSON file.

    Args:
        filepath (str) : The path to the JSON file.

    Returns:
        dict : The data from the JSON file.
    """
    with open(file=filepath, mode="r", encoding="utf-8") as file_buffer:
        return json.load(fp=file_buffer)

# === CSV ===
def read_text_from_csv_file(filepath, title = False):
    """
    Read the text data from CSV file.

    Args:
        filepath (str) : The path to the CSV file.
        title (bool) : Whether to include the title or not. The `title` column must exists.

    Returns:
        list[[str | None, str]] : The list of text data from CSV file.
    """
    with open(file=filepath, mode="r", encoding="utf-8") as file_buffer:
        reader = csv.reader(file_buffer)

        # checking columns
        columns = next(reader)
        text_index = columns.index("text")

        if title:
            title_index = columns.index("title")

        dataset = []
        for row in reader:
            dataset.append([row[title_index] if title else None, row[text_index]])

    return dataset
