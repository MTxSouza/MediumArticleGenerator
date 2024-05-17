import argparse
import csv
import re

import numpy as np
import tqdm

from dev.utils.data import remove_links, remove_non_utf_8
from dev.utils.file import create_directory, save_numpy_file
from model.tokenizer import GPTTokenizer


def _arguments():
    """Parse the arguments from command line."""
    parser = argparse.ArgumentParser(description="Tokenize the text dataset for training the model.")
    parser.add_argument("--file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--drop-long", type=int, default=600, help="Drop articles with more than N tokens.")
    parser.add_argument("--drop-short", type=int, default=100, help="Drop articles with less than N tokens.")
    parser.add_argument("--header", action="store_true", help="If the CSV file has a header row.")
    return parser.parse_args()


def main():
    """Tokenize the text dataset and save it as compressed numpy file."""
    args = _arguments()

    # Loading Bert Tokenizer
    tokenizer = GPTTokenizer()

    # Define empty variables to store tokens and metadata
    samples = []
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
            text = remove_non_utf_8(text=text)
            text = remove_links(text=text)

            # Removing double break lines
            text = re.sub(pattern="\n\n+", repl="\n", string=text)

            # Skip empty articles
            try:
                title, article = text.split(sep="<<|custom-separator|>>")
            except ValueError:
                continue

            title = title.strip()
            article = article.strip()
            if not article or not title:
                continue

            # Tokenize the text
            sample = GPTTokenizer.SOT + title + "\n\n" + article + GPTTokenizer.EOA
            article_tokens = tokenizer.encode(text=sample)

            # Filter articles based on length
            article_length = len(article_tokens)
            if article_length >= args.drop_short and article_length <= args.drop_long:
                samples.append(sample)  # Add tokens to the main list

                if article_length > longest_article:
                    longest_article = article_length
                elif article_length < shortest_article:
                    shortest_article = article_length

        print(f"Number of articles: {len(samples)}")
        print(f"Longest article: {longest_article} tokens.")
        print(f"Shortest article: {shortest_article} tokens.")

    # Convert tokens into integers
    indices = []
    for sample in tqdm.tqdm(iterable=samples, desc="Converting tokens into integers..."):
        indices.append(tokenizer.encode(text=sample))

    # Padding the tokens to the same length
    for token_seq in tqdm.tqdm(iterable=indices, desc=f"Padding the tokens to {longest_article} tokens..."):
        token_seq.extend([tokenizer.pad_index] * (longest_article - len(token_seq)))

    # Cast the tokens to numpy array
    numpy_tokens = np.asarray(a=indices, dtype=np.int32)

    # Saving results
    print("Saving the results...")
    create_directory(directory="source")
    save_numpy_file(filepath="./source/tokens.npz", data=numpy_tokens)

    print("Done!")


if __name__ == "__main__":
    main()
