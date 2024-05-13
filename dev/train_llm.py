"""
Load the tokenized text and the vocabulary and train the LLM model. It expects the tokenized text as
a compressed numpy file and the vocabulary as a JSON file in `source` directory.

All training status will be saved in Weight & Biases platform and it requires an account to use it.

For more information use --help command.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch.utils.data import DataLoader

from dev.utils.data import ArticleDataset, get_device, split_data
from dev.utils.file import load_json_file, load_numpy_file
from model import ArticleGenerator
from model.tokenizer import Tokenizer


def _arguments():
    """Parse the arguments from command line."""
    parser = argparse.ArgumentParser(description="Train the LLM model.")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of layers for the model.")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of heads for the model.")
    parser.add_argument("--d-ff", type=int, default=1024, help="Feed-forward dimension for the model.")
    parser.add_argument("--d-model", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--context-size", type=int, default=128, help="Context size for the model.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for the model.")
    parser.add_argument("--train-size", type=float, default=0.8, help="Percentage of the dataset to use for training.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs for training.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for training.")
    parser.add_argument("--early-stop", type=int, default=10, help="Number of epochs to wait for early stopping.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    return parser.parse_args()

def model_metric(yhat, y, tokenizer):
    """
    Model metric for the LLM model.

    Args:
        - yhat (torch.Tensor) : The predicted values.
        - y (torch.Tensor) : The true values.
        - tokenizer (Tokenizer) : The tokenizer for the model.

    Returns:
        - tuple[torch.Tensor, float] : The loss and the accuracy.
    """
    batch_size, ctx, _ = yhat.size()

    base_yhat = yhat.view(batch_size * ctx, -1)
    base_y = y.view(-1)

    loss = F.cross_entropy(input=base_yhat, target=base_y, ignore_index=tokenizer.pad_index)

    pred = yhat.argmax(dim=-1)
    pred_tokens = [tokenizer.decode(tokens=tokens, apply_join=False) for tokens in pred.tolist()]
    true_tokens = [tokenizer.decode(tokens=tokens, apply_join=False) for tokens in y.tolist()]

    accuracies = [sentence_bleu(references=[true], hypothesis=pred, smoothing_function=SmoothingFunction().method1) for pred, true in zip(pred_tokens, true_tokens)]
    acc = sum(accuracies) / batch_size

    return loss, acc


def train(model, train_loader, valid_loader, optimizer, tokenizer, device, **params):
    """
    Train the LLM model.

    Args:
        - model (nn.Module) : The LLM model.
        - train_loader (DataLoader) : The training dataloader.
        - valid_loader (DataLoader) : The validation dataloader.
        - optimizer (torch.optim.Optimizer) : The optimizer for training.
        - tokenizer (Tokenizer) : The tokenizer for the model.
        - device (torch.device) : The device for the model.
    """
    # constants
    train_size = len(train_loader)
    valid_size = len(valid_loader)

    curr_iter = 1
    last_save = 0
    best_loss = np.inf

    last_train_loss = 0
    last_valid_loss = 0
    overfit = 0

    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    # wandb init
    print("=" * 100)
    run = wandb.init(project="Medium Article Generator", config=params)
    run.log_artifact("./source/vocab.json")
    run.log_artifact("./source/mapper.json")

    # training loop
    print("=" * 100)
    print("Training the LLM model...")

    for curr_epoch in range(1, params.get("epochs") + 1):

        print("EPOCH {0}/{1}".format(curr_epoch, params.get("epochs")))
        print("Overfit {0}/{1} | Best valid loss {2} | Last save : {3}".format(
            overfit,
            params.get("early_stop"),
            best_loss, last_save
        ))
        if overfit == params.get("early_stop"):
            break

        model.train()
        train_loss = 0
        train_acc = 0
        train_tqdm = tqdm.tqdm(iterable=train_loader)
        for x, y in train_tqdm:

            x, y = x.to(device=device), y.to(device=device)
            yhat = model(x)

            loss, acc = model_metric(yhat=yhat, y=y, tokenizer=tokenizer)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            curr_loss = loss.cpu().item()
            train_losses.append(curr_loss)
            train_accuracies.append(acc)
            train_loss += curr_loss
            train_acc += acc

            train_tqdm.set_description(desc=f"loss : {curr_loss} - accuracy : {acc}")
        train_loss /= train_size
        train_acc /= train_size
        print(f"[train] loss : {train_loss} - accuracy : {train_acc}")

        model.eval()
        valid_tqdm = tqdm.tqdm(iterable=valid_loader)
        valid_loss = 0
        valid_acc = 0
        with torch.no_grad():
            for x, y in valid_tqdm:

                x, y = x.to(device=device), y.to(device=device)
                yhat = model(x)

                loss, acc = model_metric(yhat=yhat, y=y, tokenizer=tokenizer)

                curr_loss = loss.cpu().item()
                valid_losses.append(curr_loss)
                valid_accuracies.append(acc)
                valid_loss += curr_loss
                valid_acc += acc

                valid_tqdm.set_description(desc=f"loss : {curr_loss} - accuracy : {acc}")
            valid_loss /= valid_size
            valid_acc /= valid_size
            print(f"[valid] loss : {valid_loss} - accuracy : {valid_acc}")

        if best_loss > valid_loss:
            best_loss = valid_loss
            last_save = curr_iter
            overfitting = 0
            torch.save(obj=model.module.state_dict(), f="./source/weights.pt")
            run.log_model(path="./source/weights.pt", name="medium_article_generator_model")
        elif last_train_loss > train_loss and valid_loss > last_valid_loss:
            overfitting += 1
        elif overfitting:
            overfitting -= 1

        curr_iter += 1
        last_train_loss = train_loss
        last_valid_loss = valid_loss

        run.log({
                "train_accuracy": train_acc,
                "train_loss": train_loss,
                "valid_accuracy": valid_acc,
                "valid_loss": valid_loss,
        }, step=curr_epoch, commit=True)
        print("=" * 100)

    print("Training completed.")


def main():
    """Train the LLM model."""
    args = _arguments()

    # loading the tokenized text
    print("=" * 100)
    print("Loading the tokenized text...")
    tokens = load_numpy_file(filepath="./source/tokens.npz")
    context_size = tokens.shape[1]

    # loading the vocabulary
    print("Loading the vocabulary...")
    vocab = load_json_file(filepath="./source/vocab.json")

    # loading Tokenizer
    print("Loading the Tokenizer...")
    tokenizer = Tokenizer(vocab=vocab)

    # preparing dataset
    print("-" * 100)
    print("Splitting the dataset...")
    assert args.train_size > 0 and args.train_size < 1, "The train size must be between 0 and 1."
    train_tokens, valid_tokens = split_data(data=tokens, train_size=args.train_size, seed=args.seed)

    # creating the dataset
    print("Creating the dataset...")
    train_dataset = ArticleDataset(articles=train_tokens)
    valid_dataset = ArticleDataset(articles=valid_tokens)

    # creating the dataloader
    print("Creating the dataloader...")
    assert args.batch_size > 0, "The batch size must be bigger than 0."
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size)

    # device
    print("-" * 100)
    device = get_device()
    print("Device: ", device.type)

    # instantiating the model
    print("Instantiating the model...")
    assert args.n_layers > 0, "The number of layers must be bigger than 0."
    assert args.n_heads > 0, "The number of heads must be bigger than 0."
    assert args.d_ff > 0, "The feed-forward dimension must be bigger than 0."
    assert args.d_model > 0, "The embedding dimension must be bigger than 0."
    assert args.dropout > 0 and args.dropout < 1, "The dropout rate must be between 0 and 1."

    assert args.d_model % args.n_heads == 0, "The embedding dimension must be divisible by the number of heads."
    model = ArticleGenerator(
        vocab_size=len(tokenizer),
        emb_dim=args.d_model,
        n_layers=args.n_layers,
        head_dim=args.d_model // args.n_heads,
        ff_dim=args.d_ff,
        context=context_size,
        dropout_rate=args.dropout,
        tokenizer=tokenizer,
        device=device
    )
    model = nn.DataParallel(module=model) # multiple GPUs

    # initializing weights
    print("Initializing the model weights...")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    def init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(tensor=module.weight, generator=generator)
            if module.bias is not None:
                nn.init.zeros_(tensor=module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(tensor=module.weight, generator=generator)
    model.apply(fn=init_weights)
    model.to(device=device)

    # optimizer
    print("Creating the optimizer...")
    assert args.lr > 0, "The learning rate must be bigger than 0."
    assert args.weight_decay >= 0, "The weight decay must be bigger or equal to 0."
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training
    train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        tokenizer=tokenizer,
        device=device,
        **vars(args)
    )


if __name__=="__main__":
    main()
