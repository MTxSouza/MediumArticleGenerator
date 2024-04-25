import tiktoken


class TikTokenizer:

    cl100k_base = tiktoken.get_encoding(encoding_name="cl100k_base")
    SOS = "<|sos|>"
    EOS = "<|eos|>"
    UNK = "<|unk|>"
    enc = tiktoken.Encoding(
        name="cl100k_im",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            SOS: 100264,
            EOS: 100265,
            UNK: 100266,
        }
    )

    @classmethod
    def vocab_size(cls) -> int:
        """Return the size of vocabulary."""
        return cls.enc.max_token_value

    @classmethod
    def encode(cls, text: str) -> list[int]:
        """Convert a sequence of characters into a sequence of numbers/indices."""
        return cls.enc.encode(text=text, allowed_special={cls.SOS, cls.EOS, cls.UNK})

    @classmethod
    def decode(cls, tokens: list[int]) -> str:
        """Convert a sequence of numbers/indices into a sequence of characters."""
        return cls.enc.decode(tokens=tokens)


class Tokenizer:

    def __init__(self, vocab: dict[int, str], lookup_vocab: dict[int, int]) -> None:
        self.vocab = {int(k): v for k, v in vocab.items()}
        self.lookup = {int(k): int(v) for k, v in lookup_vocab.items()}
        self.unk = self.lookup.get(TikTokenizer.encode(text=TikTokenizer.UNK)[0])

    def __len__(self) -> int:
        """Return the size of vocabulary."""
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        """Convert a sequence of characters into a sequence of numbers/indices."""
        tik_tokens = TikTokenizer.encode(text=text)
        custom_tokens = [self.lookup.get(tk, self.unk) for tk in tik_tokens]
        return custom_tokens

    def decode(self, tokens: list[int], apply_join: bool = True) -> str:
        """Convert a sequence of numbers/indices into a sequence of characters."""
        string = [self.vocab.get(tk) for tk in tokens]
        if apply_join:
            return "".join(string)
        return string
