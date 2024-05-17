from pydantic import BaseModel


class Prompt(BaseModel):
    text: str = ""
    extra_tokens: int = 50
    max_length: int | None = 80
    temp: float = 0.0
