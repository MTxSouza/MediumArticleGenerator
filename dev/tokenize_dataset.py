"""
Load the text dataset and tokenize it for training the model. It expects a CSV file with the a column
called `text` with the text data. It will tokenize the text and save it as compressed numpy file.
It uses the `tiktoken` library by OpenAI and you can choose the tokenizer model to use.

For more information use --help command.
"""
import argparse
import csv

import numpy as np
import tqdm

from dev.utils.data import remove_links, remove_non_utf_8
from dev.utils.file import create_directory, save_json_file, save_numpy_file
# from model.tokenizer import TikTokenizer
from model.tokenizer import Tokenizer


def _arguments():
  """Parse the arguments from command line."""
  parser = argparse.ArgumentParser(description="Tokenize the text dataset for training the model.")
  parser.add_argument("--file", type=str, required=True, help="Path to the input CSV file.")
  parser.add_argument("--only-utf-8", action="store_true", help="Only consider the UTF-8 characters.")
  parser.add_argument("--lower", action="store_true", help="Convert text to lower case.")
  parser.add_argument("--drop-long", type=int, default=600, help="Drop articles with more than N tokens.")
  parser.add_argument("--drop-short", type=int, default=100, help="Drop articles with less than N tokens.")
  parser.add_argument("--no-double-bl", action="store_true", help="Remove double break lines.")
  parser.add_argument("--header", action="store_true", help="If the CSV file has a header row.")
  return parser.parse_args()


def main():
    """Tokenize the text dataset and save it as compressed numpy file."""
    args = _arguments()

    # Define empty variables to store tokens and metadata
    tokens = []
    custom_vocab = {}
    longest_article = 0
    shortest_article = float("inf")

    # Open the CSV file
    with open(file=args.file, mode="r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)

        # Skip header row if it exists
        if args.header:
            next(reader, None)

        for row in tqdm.tqdm(iterable=reader, desc="Processing dataset..."):

            # Retrieve the title and article from the row
            title = row[0]
            article = row[1]

            # Removing white spaces from the start and end of the string
            title = title.strip()
            article = article.strip()
            if not article or not title:
                continue

            # Preprocess text (remove links, non-UTF-8 chars)
            text = title + "<<|custom-separator|>>" + article
            if args.only_utf_8:
                text = remove_non_utf_8(text=text)
            text = remove_links(text=text)

            # Removing double break lines
            if args.no_double_bl:
                text = text.replace("\n\n", "\n")

            # Convert text to lower case
            if args.lower:
                text = text.lower()

            # Skip empty articles
            try:
                title, article = text.split(sep="<<|custom-separator|>>")
            except ValueError:
                continue

            title = title.strip()
            article = article.strip()
            if not article or not title:
                continue

            # Adding special tokens to the text
            text = title + " " + article

            # Tokenize the text
            article_tokens = Tokenizer.get_str_tokens(text=Tokenizer.SOT + \
                                                            title + \
                                                            Tokenizer.EOT + \
                                                            Tokenizer.SOA + \
                                                            article + \
                                                            Tokenizer.EOA
            )

            # Filter articles based on length
            article_length = len(article_tokens)
            if article_length >= args.drop_short and article_length <= args.drop_long:
                tokens.append(article_tokens)  # Add tokens to the main list

                if article_length > longest_article:
                    longest_article = article_length
                elif article_length < shortest_article:
                    shortest_article = article_length

                # Saving into vocabulary
                for token in list(set(Tokenizer.get_str_tokens(text=title + "\n" + article))):
                    custom_vocab[token] = custom_vocab.get(token, len(custom_vocab))

        print(f"Patterner used: {Tokenizer.pattern_}")
        print(f"Number of articles: {len(tokens)}")
        print(f"Longest article: {longest_article} tokens.")
        print(f"Shortest article: {shortest_article} tokens.")
        print(f"Vocabulary size: {len(custom_vocab)}")

    # Adding special tokens to the vocabulary
    print("Adding special tokens to the vocabulary...")
    for token in Tokenizer.special_tokens_:
        custom_vocab[token] = custom_vocab.get(token, len(custom_vocab))
        print(f"\t- Adding token: {token} -> {custom_vocab[token]}")
    print(f"New vocabulary size: {len(custom_vocab)}")

    # Convert tokens into integers
    # print("Converting tokens into integers...")
    indices = []
    for token_seq in tqdm.tqdm(iterable=tokens, desc="Converting tokens into integers..."):
        indices.append([])
        for token in token_seq:
            indices[-1].append(custom_vocab[token])

    # Padding the tokens to the same length
    for token_seq in tqdm.tqdm(iterable=indices, desc=f"Padding the tokens to {longest_article} tokens..."):
        token_seq.extend([custom_vocab[Tokenizer.PAD]] * (longest_article - len(token_seq)))

    # Cast the tokens to numpy array
    numpy_tokens = np.asarray(a=indices, dtype=np.int32)

    # Saving results
    print("Saving the results...")
    create_directory(directory="source")
    save_json_file(filepath="./source/vocab.json", data=custom_vocab)
    save_numpy_file(filepath="./source/tokens.npz", data=numpy_tokens)

    print("Done!")


if __name__ == "__main__":
    main()
