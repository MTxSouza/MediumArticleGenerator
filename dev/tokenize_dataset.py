"""
Load the text dataset and tokenize it for training the model. It expects a CSV file with the a column
called `text` with the text data. It will tokenize the text and save it as compressed numpy file.
It uses the `tiktoken` library by OpenAI and you can choose the tokenizer model to use.

For more information use --help command.
"""
import argparse
import csv

import tqdm

from dev.utils.data import remove_links, remove_non_utf_8
from dev.utils.file import create_directory, save_json_file, save_numpy_file
from model.tokenizer import TikTokenizer


def _arguments():
  """Parse the arguments from command line."""
  parser = argparse.ArgumentParser(description="Tokenize the text dataset for training the model.")
  parser.add_argument("--file", type=str, required=True, help="Path to the input CSV file.")
  parser.add_argument("--only-utf-8", action="store_true", help="Only consider the UTF-8 characters.")
  parser.add_argument("--drop-long", type=int, default=600, help="Drop articles with more than N tokens.")
  parser.add_argument("--header", action="store_true", help="If the CSV file has a header row.")
  return parser.parse_args()


def main():
    """Tokenize the text dataset and save it as compressed numpy file."""
    args = _arguments()

    # Define empty variables to store tokens and metadata
    tokens = []
    custom_vocab = {}
    vocab_mapper = {}
    longest_article = 0
    shortest_article = float("inf")

    # Open the CSV file
    with open(file=args.file, mode="r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)

        # Skip header row if it exists
        if args.header:
            next(reader, None)

        for title, article in tqdm.tqdm(iterable=reader, desc="Processing dataset..."):

            # removing white spaces from the start and end of the string
            title = title.strip()
            article = article.strip()
            if not article or not title:
                continue

            # Preprocess text (remove links, non-UTF-8 chars)
            text = title + "<<|custom-separator|>>" + article
            if args.only_utf_8:
                text = remove_non_utf_8(text=text)
            text = remove_links(text=text)

            # Skip empty articles
            title, article = text.split(sep="<<|custom-separator|>>")
            title = title.strip()
            article = article.strip()
            if not article or not title:
                continue

            # Adding special tokens to the text
            text = TikTokenizer.SOT + title + TikTokenizer.EOT + TikTokenizer.SOA + article + TikTokenizer.EOA

            # Tokenize the text
            article_tokens = TikTokenizer.encode(text=text)

            # Filter articles based on length
            if len(article_tokens) <= args.drop_long:
                tokens.extend(article_tokens)  # Add tokens to the main list

                if len(article_tokens) > longest_article:
                    longest_article = len(article_tokens)
                elif len(article_tokens) < shortest_article:
                    shortest_article = len(article_tokens)
        print(f"Number of articles: {len(tokens)}")
        print(f"Longest article: {longest_article} tokens.")
        print(f"Shortest article: {shortest_article} tokens.")

    # Create vocabulary and tokenizer after processing the entire dataset
    print("Retrieving unique tokens...")
    unique_tokens = sorted(list(set(tokens + TikTokenizer.encode(text=TikTokenizer.UNK))))
    for token in tqdm.tqdm(iterable=unique_tokens, desc="Creating vocabulary..."):
        str_token = TikTokenizer.decode(tokens=[token])
        idx = len(custom_vocab)
        custom_vocab[idx] = str_token
        vocab_mapper[token] = idx

    # Remap the tokens to the new vocabulary
    print("Remapping the tokens...")
    tokens = list(map(vocab_mapper.get, tokens))

    # Saving results
    print("Saving the results...")
    create_directory(directory="source")
    save_json_file(filepath="./source/vocab.json", data=custom_vocab)
    save_json_file(filepath="./source/mapper.json", data=vocab_mapper)
    save_numpy_file(filepath="./source/tokens.npz", data=tokens)

    print("Done!")


if __name__ == "__main__":
    main()
