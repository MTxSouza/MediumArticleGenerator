import torch
import torch.nn as nn

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


class BertEmbedding(nn.Module):

    def __init__(self):
        """
        Bert embedding layer for the transformer model.
        """
        super().__init__()
        self.embedding = torch.hub.load("huggingface/pytorch-transformers", "model", "bert-base-cased")
        # Freeze the parameters
        for param in self.embedding.parameters():
            param.requires_grad = False

    @property
    def embedding_dim(self):
        """
        Get the embedding dimension.

        Returns:
            int : The embedding dimension.
        """
        return 768

    def forward(self, x):
        """
        Forward pass of the Bert embedding layer.

        Args:
            x (torch.Tensor) : The input tensor.

        Returns:
            torch.Tensor : The output tensor.
        """
        model_logger.debug("BertEmbedding: x shape: %s", x.size())

        # Get mask for the input tensor
        mask = (x != 0).float() # 0 is the padding token

        out = self.embedding(x, mask)
        emb = out.last_hidden_state
        model_logger.debug("BertEmbedding: emb shape: %s", emb.size())
        return emb
