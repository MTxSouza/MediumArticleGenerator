import tiktoken


class TikTokenizer:

    cl100k_base = tiktoken.get_encoding(encoding_name="cl100k_base")
    SOT = "<|sot|>" # start of title
    EOT = "<|eot|>" # end of title
    SOA = "<|soa|>" # start of article
    EOA = "<|eoa|>" # end of article
    UNK = "<|unk|>" # unknown token
    PAD = "<|pad|>"

    __current_vocab_size = cl100k_base.n_vocab
    _new_special_tokens = {}
    for __token in (SOT, EOT, SOA, EOA, UNK, PAD):
        _new_special_tokens[__token] = __current_vocab_size
        __current_vocab_size += 1

    enc = tiktoken.Encoding(
        name="cl100k_im",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            **_new_special_tokens
        }
    )

    INDEX_SOT = _new_special_tokens.get(SOT)
    INDEX_EOT = _new_special_tokens.get(EOT)
    INDEX_SOA = _new_special_tokens.get(SOA)
    INDEX_EOA = _new_special_tokens.get(EOA)
    INDEX_UNK = _new_special_tokens.get(UNK)
    INDEX_PAD = _new_special_tokens.get(PAD)

    @staticmethod
    def list_models():
        """
        Return the list of available models.

        Returns:
            List[str]: The list of available models.
        """
        return tiktoken.list_encoding_names()

    @classmethod
    def vocab_size(cls):
        """
        Return the size of vocabulary.

        Returns:
            int: The size of vocabulary.
        """
        return cls.enc.max_token_value

    @classmethod
    def encode(cls, text):
        """
        Convert a sequence of characters into a sequence of numbers/indices.

        Args:
            text (str) : The input text.

        Returns:
            List[int] : The list of numbers/indices.
        """
        return cls.enc.encode(text=text, allowed_special=set(
            list(cls._new_special_tokens) + \
            list(cls.cl100k_base._special_tokens)
        ))

    @classmethod
    def decode(cls, tokens):
        """
        Convert a sequence of numbers/indices into a sequence of characters.

        Args:
            tokens (List[int]) : The list of numbers/indices.

        Returns:
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

        # special tokens
        self._special_tokens = {
            tk: self.lookup.get(TikTokenizer.encode(text=tk)[0])
            for tk, _ in TikTokenizer._new_special_tokens.items()
        }

    @property
    def pad_index(self):
        """
        Return the index of padding token.

        Returns:
            int: The index of padding token.
        """
        return self._special_tokens.get("<|pad|>")

    def __len__(self):
        """
        Return the size of vocabulary.

        Returns:
            int: The size of vocabulary.
        """
        return len(self.vocab)

    def encode(self, text):
        """
        Convert a sequence of characters into a sequence of numbers/indices.

        Args:
            text (str) : The input text.

        Returns:
            List[int] : The list of numbers/indices.
        """
        tik_tokens = TikTokenizer.encode(text=text)
        custom_tokens = [self.lookup.get(tk, self._special_tokens.get("<|unk|>")) for tk in tik_tokens]
        return custom_tokens

    def decode(self, tokens, apply_join = True):
        """
        Convert a sequence of numbers/indices into a sequence of characters.

        Args:
            tokens (List[int]) : The list of numbers/indices.
            apply_join (bool) : Whether to join the characters or not. (default: True)

        Returns:
            str | List[str] : The output text.
        """
        string = [self.vocab.get(tk) for tk in tokens]
        if apply_join:
            return "".join(string)
        return string
