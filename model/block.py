import torch
import torch.nn as nn
import torch.nn.functional as F

from model.logger import model_logger


class FeedForward(nn.Module):

    def __init__(self, emb_dim, ff_dim, dropout_rate = 0.2):
        """
        Feed forward layer for transformer. Consists of two linear layers with a ReLU activation function.

        Args:
            emb_dim (int) : The dimension of the embedding.
            ff_dim (int) : The dimension of the feed forward layer.
            dropout_rate (float) : The dropout rate. (default: 0.2)
        """
        super().__init__()
        self.linear_1 = nn.Linear(in_features=emb_dim, out_features=ff_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear_2 = nn.Linear(in_features=ff_dim, out_features=emb_dim)

    def forward(self, x):
        """
        Forward pass of the feed forward layer.

        Args:
            x (torch.Tensor) : The input tensor.

        Returns:
            torch.Tensor : The output tensor.
        """
        model_logger.debug("Feed forward: x shape: %s", x.size())
        z1 = self.linear_1(x)
        model_logger.debug("Feed forward: z1 shape: %s", z1.size())
        a1 = self.relu(z1)
        a1 = self.dropout(a1)
        z2 = self.linear_2(a1)
        model_logger.debug("Feed forward: z2 shape: %s", z2.size())
        return z2


class Attention(nn.Module):

    def __init__(self, emb_dim, head_dim, context, dropout_rate):
        """
        Attention layer for transformer. Computes the attention weights and applies them to the input tensor.

        Args:
            emb_dim (int) : The dimension of the embedding.
            head_dim (int) : The dimension of the attention head.
            context (int) : The maximum length of the sequence.
            dropout_rate (float) : The dropout rate.
        """
        super().__init__()

        self.query = nn.Linear(in_features=emb_dim, out_features=head_dim, bias=False)
        self.key = nn.Linear(in_features=emb_dim, out_features=head_dim, bias=False)
        self.value = nn.Linear(in_features=emb_dim, out_features=head_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)

        ones = torch.ones(size=[context, context], dtype=torch.float)
        self.register_buffer(name="mask", tensor=torch.tril(input=ones))

    def forward(self, x):
        """
        Forward pass of the attention layer.

        Args:
            x (torch.Tensor) : The input tensor.

        Returns:
            torch.Tensor : The output tensor.
        """
        model_logger.debug("Attention: x shape: %s", x.size())
        B, T, C = x.size()
        Q = self.query(x)
        model_logger.debug("Attention: Q shape: %s", Q.size())
        K = self.key(x)
        model_logger.debug("Attention: K shape: %s", K.size())
        V = self.value(x)
        model_logger.debug("Attention: V shape: %s", V.size())

        QK = Q @ K.transpose(-2, -1) * C**-0.5
        model_logger.debug("Attention: QK shape: %s", QK.size())
        model_logger.debug("Attention: mask shape: %s", self.mask.size())
        attention = QK.masked_fill(self.mask[:T,:T] == 0, float("-inf"))
        attention = F.softmax(input=attention, dim=-1)

        attention = self.dropout(attention)

        out = attention @ V
        model_logger.debug("Attention: out shape: %s", out.size())

        return out


class MultiAttention(nn.Module):

    def __init__(self, emb_dim, head_dim, context, dropout_rate):
        """
        Multi-head attention layer for transformer. Computes multiple attention heads and concatenates them.

        Args:
            emb_dim (int) : The dimension of the embedding.
            head_dim (int) : The dimension of the attention head.
            context (int) : The maximum length of the sequence.
            dropout_rate (float) : The dropout rate.
        """
        super().__init__()
        n_heads = emb_dim // head_dim
        self.attention = nn.ModuleList(modules=[Attention(emb_dim=emb_dim, head_dim=head_dim, context=context, dropout_rate=dropout_rate) for _ in range(n_heads)])
        self.linear = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """
        Forward pass of the multi-head attention layer.

        Args:
            x (torch.Tensor) : The input tensor.

        Returns:
            torch.Tensor : The output tensor.
        """
        model_logger.debug("Multi-attention: x shape: %s", x.size())
        out = torch.cat(tensors=[attention(x) for attention in self.attention], dim=-1)
        model_logger.debug("Multi-attention: out shape: %s", out.size())
        out = self.linear(out)
        model_logger.debug("Multi-attention: out shape: %s", out.size())
        out = self.dropout(out)
        return out


class DecoderLayer(nn.Module):

    def __init__(self, emb_dim, head_dim, context, ff_dim, dropout_rate):
        """
        Decoder layer for transformer. Consists of a multi-head attention layer and a feed forward layer.

        Args:
            emb_dim (int) : The dimension of the embedding.
            head_dim (int) : The dimension of the attention head.
            context (int) : The maximum length of the sequence.
            ff_dim (int) : The dimension of the feed forward layer.
            dropout_rate (float) : The dropout rate.
        """
        super().__init__()
        self.attention = MultiAttention(emb_dim=emb_dim, head_dim=head_dim, context=context, dropout_rate=dropout_rate)
        self.feed_forward = FeedForward(emb_dim=emb_dim, ff_dim=ff_dim, dropout_rate=dropout_rate)
        self.norm_1 = nn.LayerNorm(normalized_shape=emb_dim)
        self.norm_2 = nn.LayerNorm(normalized_shape=emb_dim)

    def forward(self, x):
        """
        Forward pass of the decoder layer.

        Args:
            x (torch.Tensor) : The input tensor.

        Returns:
            torch.Tensor : The output tensor.
        """
        model_logger.debug("Decoder layer: x shape: %s", x.size())
        x_norm = self.norm_1(x)
        attention = self.attention(x_norm)
        model_logger.debug("Decoder layer: attention shape: %s", attention.size())
        attention = attention + x

        attention_norm = self.norm_2(attention)
        ff = self.feed_forward(attention_norm)
        model_logger.debug("Decoder layer: ff shape: %s", ff.size())
        ff = ff + attention

        return ff


class Decoder(nn.Module):

    def __init__(self, n_layers, decoder):
        """
        Decoder for transformer. Consists of multiple decoder layers.

        Args:
            n_layers (int) : The number of decoder layers.
            decoder (DecoderLayer) : The decoder layer.
        """
        super().__init__()
        self.layers = nn.Sequential(*[decoder for _ in range(n_layers)])

    def forward(self, x):
        """
        Forward pass of the decoder.

        Args:
            x (torch.Tensor) : The input tensor.

        Returns:
            torch.Tensor : The output tensor.
        """
        model_logger.debug("Decoder: x shape: %s", x.size())
        return self.layers(x)
