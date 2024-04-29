from enum import Enum

from pydantic import BaseModel


class SourceData(Enum):
    model = "weights.pt"
    dataset = "medium_articles.csv"


class ModelDetails(BaseModel):
    num_layers: int
    embedding_dim: int
    head_dim: int
    feed_forward_dim: int
    context_window: int
