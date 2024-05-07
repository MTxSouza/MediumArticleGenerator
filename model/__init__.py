"""
This module provides the entire implementation of Medium Article Generator model.
"""
import torch
import torch.nn as nn

from model.nn import DecoderLayer, PositionalEncoding


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
            tokenizer (Tokenizer) : The tokenizer object.
        """
        super().__init__()
        self.ctx = context
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.pe = PositionalEncoding(context=context, emb_dim=emb_dim)
        self.layers = nn.Sequential(*[DecoderLayer(emb_dim=emb_dim, head_dim=head_dim, context=context, ff_dim=ff_dim, dropout_rate=dropout_rate) for _ in range(n_layers)])
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
        x = self.pe(x)
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

    def generate(self, text = "<|sos|>", max_len = None):
        """
        Generate text based on the input text.
        Args:
            text (str) : The input text. (default: "<|sos|>")
            max_len (int) : The maximum length of the generated text. (default: None)
        Output:
            Generator[str, None, None] : The generated text.
        """
        max_len = float("inf") if max_len is None else max_len
        count = 0
        x = torch.tensor(data=self.tokenizer.encode(text=text), requires_grad=False).unsqueeze(dim=0).to(device=self.dev)
        assert x.ndim == 2
        yield text
        while count < max_len:
            if x.size(dim=1) > self.ctx:
                x = x[:,1:] # ignoring first token of window context
            token = self.predict_next_token(x)
            if token.item() == self.tokenizer.eos: # end of sentence
                break
            yield self.tokenizer.decode(tokens=[token.item()])
            x = torch.cat(tensors=[x, token.unsqueeze(dim=0)], dim=1)
            count += 1
