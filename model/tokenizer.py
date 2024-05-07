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

    @staticmethod
    def list_models():
        """
        Return the list of available models.
        Output:
            List[str]: The list of available models.
        """
        return tiktoken.list_encoding_names()

    @classmethod
    def vocab_size(cls):
        """
        Return the size of vocabulary.
        Output:
            int: The size of vocabulary.
        """
        return cls.enc.max_token_value

    @classmethod
    def encode(cls, text):
        """
        Convert a sequence of characters into a sequence of numbers/indices.
        Args:
            text (str) : The input text.
        Output:
            List[int] : The list of numbers/indices.
        """
        return cls.enc.encode(text=text, allowed_special={cls.SOS, cls.EOS, cls.UNK})

    @classmethod
    def decode(cls, tokens):
        """
        Convert a sequence of numbers/indices into a sequence of characters.
        Args:
            tokens (List[int]) : The list of numbers/indices.
        Output:
            str : The output text.
        """
        return cls.enc.decode(tokens=tokens)


class Tokenizer:

    def __init__(self, vocab, lookup_vocab):
        """
        Tokenizer class for converting text to numbers and vice versa.
        Args:
            vocab (Dict[int, str]) : The vocabulary dictionary.
            lookup_vocab (Dict[int, int]) : The lookup vocabulary dictionary.
        """
        self.vocab = {int(k): v for k, v in vocab.items()}
        self.lookup = {int(k): int(v) for k, v in lookup_vocab.items()}
        self.unk = self.lookup.get(TikTokenizer.encode(text=TikTokenizer.UNK)[0])
        self.eos = self.lookup.get(TikTokenizer.encode(text=TikTokenizer.EOS)[0])

    def __len__(self):
        """
        Return the size of vocabulary.
        Output:
            int: The size of vocabulary.
        """
        return len(self.vocab)

    def encode(self, text):
        """
        Convert a sequence of characters into a sequence of numbers/indices.
        Args:
            text (str) : The input text.
        Output:
            List[int] : The list of numbers/indices.
        """
        tik_tokens = TikTokenizer.encode(text=text)
        custom_tokens = [self.lookup.get(tk, self.unk) for tk in tik_tokens]
        return custom_tokens

    def decode(self, tokens, apply_join = True):
        """
        Convert a sequence of numbers/indices into a sequence of characters.
        Args:
            tokens (List[int]) : The list of numbers/indices.
            apply_join (bool) : Whether to join the characters or not. (default: True)
        Output:
            str | List[str] : The output text.
        """
        string = [self.vocab.get(tk) for tk in tokens]
        if apply_join:
            return "".join(string)
        return string
