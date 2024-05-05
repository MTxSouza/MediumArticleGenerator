"""
Utility functions for data manipulation and processing.
"""
import csv
import json
import os
import re

import numpy as np

# global variables
_non_utf8_characters_filter = re.compile(pattern=r"[^\x00-\x7F]")
_link_pattern = re.compile(pattern=r"https?://\S+")


# === Numpy ===
def save_numpy_file(filepath, data):
    """
    Save the data as compressed numpy file.
    ---
    Params:
        - filepath (str) : The path to the compressed numpy file.
        - data (np.ndarray) : The data to save as compressed numpy file.
    """
    np.savez_compressed(file=filepath, data=data)

# === String ===
def remove_non_utf_8(text):
    """
    Remove the non-UTF-8 characters from the text.
    ---
    Params:
        - text (str) : The text to remove non-UTF-8 characters.
    Output:
        - str : The text after removing non-UTF-8 characters.
    """
    return _non_utf8_characters_filter.sub(repl="", string=text)


def remove_links(text):
    """
    Remove the links from the text.
    ---
    Params:
        - text (str) : The text to remove the links.
    Output:
        - str : The text after removing the links.
    """
    return _link_pattern.sub(repl="", string=text)

# === Directory ===
def create_directory(directory):
    """
    Create the directory if it doesn't exists.
    ---
    Params:
        - directory (str) : The path to the directory.
    """
    os.makedirs(name=directory, exist_ok=True)

# === JSON ===
def save_json_file(filepath, data):
    """
    Save the data as JSON file.
    ---
    Params:
        - filepath (str) : The path to the JSON file.
        - data (dict) : The data to save as JSON file.
    """
    with open(file=filepath, mode="w", encoding="utf-8") as file_buffer:
        json.dump(obj=data, fp=file_buffer, indent=4)

# === CSV ===
def read_text_from_csv_file(filepath, title = False):
    """
    Read the text data from CSV file.
    ---
    Params:
        - filepath (str) : The path to the CSV file.
        - title (bool) : Whether to include the title or not. The `title` column must exists.
    Output:
        - list[[str | None, str]] : The list of text data from CSV file.
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
