"""
This module provides the entire implementation of Medium Article Generator model.
"""
from typing import Generator

import torch
import torch.nn as nn

from model.blocks.nn import DecoderLayer, PositionalEncoding
from model.blocks.tokenizer import Tokenizer


class ArticleGenerator(nn.Module):

    def __init__(
            self,
            n_layers: int,
            vocab_size: int,
            emb_dim: int,
            head_dim: int,
            context: int,
            ff_dim: int,
            dropout_rate: float,
            device: str,
            tokenizer: Tokenizer
        ) -> None:
        super().__init__()
        self.ctx = context
        self.eos = vocab_size - 1
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.pe = PositionalEncoding(context=context, emb_dim=emb_dim)
        self.layers = nn.Sequential(*[DecoderLayer(emb_dim=emb_dim, head_dim=head_dim, context=context, ff_dim=ff_dim, dropout_rate=dropout_rate) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(normalized_shape=emb_dim)
        self.out = nn.Linear(in_features=emb_dim, out_features=vocab_size)

        self.dev = device
        self.tokenizer = tokenizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)
        x = self.pe(x)
        x = self.layers(x)
        x = self.norm(x)
        x = self.out(x)
        return x

    @torch.no_grad()
    def predict_next_token(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        x = self(x)
        token = x.argmax(dim=-1)[:,-1]
        return token

    def generate(self, text: str = "<|sos|>", max_len: int = None) -> Generator[str, None, None]:
        max_len = float("inf") if max_len is None else max_len
        count = 0
        x = torch.tensor(data=self.tokenizer.encode(text=text), requires_grad=False).unsqueeze(dim=0).to(device=self.dev)
        assert x.ndim == 2
        while count < max_len:
            if x.size(dim=1) > self.ctx:
                x = x[:,1:] # ignoring first token of window context
            token = self.predict_next_token(x)
            yield self.tokenizer.decode(tokens=[token.item()])
            x = torch.cat(tensors=[x, token.unsqueeze(dim=0)], dim=1)
            count += 1
