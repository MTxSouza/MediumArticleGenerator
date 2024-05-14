"""
Main script for testing the data loader and see the output token.
"""
import argparse
import os

from dev.utils.data import ChunkDataset, FullDataset
from dev.utils.file import load_json_file, load_numpy_file
from model.tokenizer import Tokenizer


def _arguments():
    """
    Parse the arguments from command line.
    """
    parser = argparse.ArgumentParser(description="Test the data loader for the model.")
    parser.add_argument("--context-size", type=int, default=0, help="Context size for the chunk dataset.")
    return parser.parse_args()


def clear_terminal():
    """
    Clear the terminal.
    """
    os.system("cls" if os.name == "nt" else "clear")


def main():
    """
    Main function to test the data loader.
    """
    # Parse the arguments
    args = _arguments()

    # Load tokens
    tokens = load_numpy_file('./source/tokens.npz')
    print("Tokens shape: ", tokens.shape)

    # Load Tokenizer
    vocab = load_json_file('./source/vocab.json')
    tokenizer = Tokenizer(vocab=vocab)
    print("Vocabulary size: ", len(tokenizer))

    # Create the dataset
    if args.context_size > 0:
        print("Creating ChunkDataset...")
        dataset = ChunkDataset(article=tokens, context_size=args.context_size)
    else:
        print("Creating FullDataset...")
        dataset = FullDataset(articles=tokens)
    print("Dataset size: ", len(dataset))

    # Get the first data
    try:
        iter_dataset = iter(dataset)
        n_iter = 0
        while True:
            print("-" * 100)
            print(f"Iteration: {n_iter}")
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
            n_iter += 1
    except KeyboardInterrupt:
        print("The data loader is stopped.")


if __name__=="__main__":
    main()
