"""
This module provides the entire implementation of Medium Article Generator model.
"""
import torch
import torch.nn as nn

from model.block import Decoder, DecoderLayer
from model.embedding import GPTEmbedding


class ArticleGenerator(nn.Module):

    def __init__(self, n_layers, vocab_size, emb_dim, head_dim, context, ff_dim, dropout_rate, device, tokenizer):
        """
        Article Generator model based on Transformer.
        Args:
            n_layers (int) : The number of decoder layers.
            vocab_size (int) : The size of vocabulary.
            emb_dim (int) : The dimension of the embedding.
            head_dim (int) : The dimension of the attention head.
            context (int) : The maximum length of the sequence.
            ff_dim (int) : The dimension of the feed forward layer.
            dropout_rate (float) : The dropout rate.
            device (str) : The device to run the model.
            tokenizer (Tokenizer | BertTokenizer) : The tokenizer object.
        """
        super().__init__()
        self.ctx = context
        self.emb = GPTEmbedding()
        self.layers = Decoder(
            n_layers=n_layers,
            decoder=DecoderLayer(
                emb_dim=emb_dim,
                head_dim=head_dim,
                context=context,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate
            )
        )
        self.norm = nn.LayerNorm(normalized_shape=emb_dim)
        self.out = nn.Linear(in_features=emb_dim, out_features=vocab_size)

        self.dev = device
        self.tokenizer = tokenizer

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor) : The input tensor.
        Output:
            torch.Tensor : The output tensor.
        """
        x = self.emb(x)
        x = self.layers(x)
        x = self.norm(x)
        x = self.out(x)
        return x

    @torch.no_grad()
    def predict_next_token(self, x):
        """
        Predict the next token given the input tensor.
        Args:
            x (torch.Tensor) : The input tensor.
        Output:
            torch.Tensor : The predicted token.
        """
        self.eval()
        x = self(x)
        token = x.argmax(dim=-1)[:,-1]
        return token

    def generate(self, text, extra_tokens = 50, max_len = None):
        """
        Generate text based on the input text.

        Args:
            text (str) : The input text.
            extra_tokens (int) : The number of extra tokens to generate when exceeding the context size. (default: 50)
            max_len (int) : The maximum number of the tokens to be generated. (default: None)

        Returns:
            Generator[str, None, None] : The generated text.
        """
        # Adding initial tags to the input text
        tagged_text = self.tokenizer.tokenize(text=text)

        # Stop token
        end_of_article = self.tokenizer.encode(text=self.tokenizer.EOA)[0]

        x = torch.tensor(
            data=tagged_text,
            requires_grad=False
        ).unsqueeze(dim=0).to(device=self.dev)
        n_tokens = x.size(dim=1) # Total number of tokens
        assert x.ndim == 2
        while n_tokens < max_len and n_tokens < self.ctx + extra_tokens:
            if x.size(dim=1) > self.ctx:
                x = x[:,1:] # ignoring first token of window context
            token = self.predict_next_token(x)
            if token.item() == end_of_article: # end of sentence
                break
            yield self.tokenizer.decode(indices=[token.item()])
            x = torch.cat(tensors=[x, token.unsqueeze(dim=0)], dim=1)
            n_tokens += 1
