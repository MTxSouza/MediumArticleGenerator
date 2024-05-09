"""
Load the text dataset and tokenize it for training the model. It expects a CSV file with the a column
called `text` with the text data. It will tokenize the text and save it as compressed numpy file.
It uses the `tiktoken` library by OpenAI and you can choose the tokenizer model to use.

For more information use --help command.
"""
import argparse

import tqdm

from dev.utils.data import remove_links, remove_non_utf_8
from dev.utils.file import (create_directory, read_text_from_csv_file,
                            save_json_file, save_numpy_file)
from model.tokenizer import TikTokenizer, Tokenizer


def _arguments():
    """Parse the arguments from command line."""
    parser = argparse.ArgumentParser(description="Tokenize the text dataset for training the model.")
    parser.add_argument("--file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--only-utf-8", action="store_true", help="Only consider the UTF-8 characters.")
    parser.add_argument("--drop-long", type=int, default=600, help="Drop articles with more than N tokens.")
    return parser.parse_args()


def main():
    """Tokenize the text dataset and save it as compressed numpy file."""
    args = _arguments()

    # loading dataset
    print("Loading the dataset...")
    dataset = read_text_from_csv_file(filepath=args.file)
    # converting into a single string
    print("Concatenating the dataset in a single string...")
    dataset_str = "".join(
        TikTokenizer.SOT + \
        title + \
        TikTokenizer.EOT + \
        TikTokenizer.SOA + \
        article + \
        TikTokenizer.EOA \
        if title else ""
        for title, article in dataset
    ).strip() # removing whitespaces in borders

    # removing non-UTF-8 characters
    if args.only_utf_8:
        print("(Optional) Removing non-UTF-8 characters...")
        dataset_str = remove_non_utf_8(text=dataset_str)

    # removing links
    print("Removing links...")
    dataset_str = remove_links(text=dataset_str)

    # creating vocabulary
    print("Tokenizing the dataset...")
    tokens = TikTokenizer.encode(text=dataset_str)

    # filtering articles
    token_chunks = []
    for token in tqdm.tqdm(iterable=tokens, desc="Filtering articles..."):
        if token == TikTokenizer.INDEX_SOT:
            token_chunks.append([]) # creating a new chunk
        token_chunks[-1].append(token)
    print(f"Number of articles: {len(token_chunks)}")

    # dropping long articles
    non_long_token_chunks = []
    longest_article = 0
    shortest_article = float("inf")
    for chunk in tqdm.tqdm(iterable=token_chunks, desc="Dropping long articles..."):
        chunk_len = len(chunk)
        if chunk_len <= args.drop_long:
            non_long_token_chunks.append(chunk)

        if chunk_len > longest_article:
            longest_article = chunk_len
        elif chunk_len < shortest_article:
            shortest_article = chunk_len
    print(f"Number of articles after dropping: {len(token_chunks)}")
    print(f"Longest article: {longest_article} tokens.")
    print(f"Shortest article: {shortest_article} tokens.")

    # joining tokens
    print("Joining tokens...")
    tokens = [token for chunk in token_chunks for token in chunk]

    print("Retrieving unique tokens...")
    unique_tokens = sorted(list(set(tokens + TikTokenizer.encode(text=TikTokenizer.UNK))))

    print("Creating vocabulary...")
    custom_vocab = {}
    vocab_mapper = {}
    for unique_tk in tqdm.tqdm(iterable=unique_tokens):
        str_token = TikTokenizer.decode(tokens=[unique_tk])
        idx = len(custom_vocab)
        custom_vocab[idx] = str_token
        vocab_mapper[unique_tk] = idx
    del unique_tokens # deleting variable

    # tokenizing the dataset
    print("Tokenizing the dataset...")
    tokenizer = Tokenizer(vocab=custom_vocab, lookup_vocab=vocab_mapper)
    tokens = tokenizer.encode(text=dataset_str)

    # saving results
    print("Saving the results...")
    create_directory(directory="source")
    save_json_file(filepath="./source/vocab.json", data=custom_vocab)
    save_json_file(filepath="./source/mapper.json", data=vocab_mapper)
    save_numpy_file(filepath="./source/tokens.npz", data=tokens)

    print("Done!")
