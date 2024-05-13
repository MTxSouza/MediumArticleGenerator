"""
Main script for testing the data loader and see the output token.
"""
import os

from dev.utils.data import ArticleDataset
from dev.utils.file import load_json_file, load_numpy_file
from model.tokenizer import Tokenizer


def clear_terminal():
    """
    Clear the terminal.
    """
    os.system("cls" if os.name == "nt" else "clear")


def main():
    """
    Main function to test the data loader.
    """
    # Load tokens
    tokens = load_numpy_file('./source/tokens.npz')
    print("Tokens shape: ", tokens.shape)

    # Load Tokenizer
    vocab = load_json_file('./source/vocab.json')
    tokenizer = Tokenizer(vocab=vocab)
    print("Vocabulary size: ", len(tokenizer))

    # Create the dataset
    dataset = ArticleDataset(articles=tokens)
    print("Dataset size: ", len(dataset))

    # Get the first data
    try:
        iter_dataset = iter(dataset)
        while True:
            print("-" * 100)
            _ = input("Press any keyboard to get the next data.")
            clear_terminal()
            x, y = next(iter_dataset)
            print("Input: ", x.size())
            print("Output: ", y.size())
            print("Token indices:")
            print("\tInput:\n", x)
            print("\n")
            print("\tOutput:\n", y)
            print("\n")
            print("Token words:")
            print("\tInput:\n", tokenizer.decode(x.numpy().tolist()))
            print("\n")
            print("\tOutput:\n", tokenizer.decode(y.numpy().tolist()))
    except KeyboardInterrupt:
        print("The data loader is stopped.")


if __name__=="__main__":
    main()
