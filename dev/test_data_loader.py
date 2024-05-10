"""
Main script for testing the data loader and see the output token.
"""
import argparse

from dev.utils.data import ArticleDataset
from dev.utils.file import load_json_file, load_numpy_file
from model.tokenizer import Tokenizer


def parse_args():
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description="Test the data loader.")
    parser.add_argument("--context", type=int, default=128, help="The context size for the data loader.")
    return parser.parse_args()


def main():
    """
    Main function to test the data loader.
    """
    # Parse the arguments
    args = parse_args()

    # Load tokens
    tokens = load_numpy_file('./source/tokens.npz')
    print("Tokens shape: ", tokens.shape)
    assert args.context > 0 and args.context < tokens.shape[1], \
        "The context size must be positive and less than the number of tokens."

    # Load Tokenizer
    vocab = load_json_file('./source/vocab.json')
    mapper = load_json_file('./source/mapper.json')
    tokenizer = Tokenizer(vocab=vocab, lookup_vocab=mapper)

    # Create the dataset
    dataset = ArticleDataset(articles=tokens, context=args.context, pad_index=tokenizer.pad_index)
    print("Dataset size: ", len(dataset))
    print("Context size: ", dataset.ctx)
    print("Limit: ", dataset.limit)

    # Get the first data
    try:
        iter_dataset = iter(dataset)
        while True:
            _ = input("Press any keyboard to get the next data.")
            x, y = next(iter_dataset)
            print("Input: ", x.size())
            print("Output: ", y.size())
            print("Input:\n", x)
            print("\n")
            print("Output:\n", y)
    except KeyboardInterrupt:
        print("The data loader is stopped.")


if __name__=="__main__":
    main()
