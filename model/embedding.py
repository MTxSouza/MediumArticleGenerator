import torch
import torch.nn as nn
from transformers import AutoModel

from model.logger import model_logger


class PositionalEncoding(nn.Module):

    def __init__(self, context, emb_dim):
        """
        Positional encoding for transformer. Adds positional information to the input tensor.

        Args:
            context (int) : The maximum length of the sequence.
            emb_dim (int) : The dimension of the embedding.
        """
        super().__init__()

        even_i = torch.arange(start=0, end=emb_dim, step=2).float()
        odd_i = torch.arange(start=1, end=emb_dim, step=2).float() - 1

        even_denom = torch.pow(10_000, exponent=even_i / emb_dim)
        odd_denom = torch.pow(10_000, exponent=odd_i / emb_dim)

        pos = torch.arange(end=context).float().reshape(shape=[context, 1])

        even = torch.sin(pos / even_denom)
        odd = torch.cos(pos / odd_denom)

        self.register_buffer(name="pe", tensor=torch.cat(tensors=[even, odd], dim=1).expand(1, -1, -1))

    def forward(self, x):
        """
        Forward pass of the positional encoding.

        Args:
            x (torch.Tensor) : The input tensor.

        Returns:
            torch.Tensor : The output tensor.
        """
        model_logger.debug("Positional encoding: x shape: %s", x.size())
        model_logger.debug("Positional encoding: pe shape: %s", self.pe.size())
        B, T, D = x.size()
        x_pe = x + self.pe[:,:T,:]
        model_logger.debug("Positional encoding: x_pe shape: %s", x_pe.size())
        return x_pe


class Embedding(nn.Module):

    def __init__(self, vocab_size, emb_dim):
        """
        Embedding layer for the transformer model.

        Args:
            vocab_size (int) : The size of the vocabulary.
            emb_dim (int) : The dimension of the embedding.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)

    @property
    def embedding_dim(self):
        """
        Get the embedding dimension.

        Returns:
            int : The embedding dimension.
        """
        return self.embedding.embedding_dim

    def forward(self, x):
        """
        Forward pass of the embedding layer.

        Args:
            x (torch.Tensor) : The input tensor.

        Returns:
            torch.Tensor : The output tensor.
        """
        model_logger.debug("Embedding: x shape: %s", x.size())
        return self.embedding(x)


class GPTEmbedding(nn.Module):

        def __init__(self):
            """
            Embedding layer for the GPT model.
            """
            super().__init__()
            model = AutoModel.from_pretrained("gpt2")
            self.embedding = model.wte
            self.pe = model.wpe

            # Freeze the parameters
            for param in self.embedding.parameters():
                param.requires_grad = False
            for param in self.pe.parameters():
                param.requires_grad = False

        def __call__(self, x):
            """
            Forward pass of the embedding layer.
    
            Args:
                x (torch.Tensor) : The input tensor.
    
            Returns:
                torch.Tensor : The output tensor.
            """
            i = x.size(1)
            emb = self.embedding(x)
            pe = self.pe(torch.arange(i).unsqueeze(0).repeat(x.size(0), 1).to(x.device))
            return emb + pe
