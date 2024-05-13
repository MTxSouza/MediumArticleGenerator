import re


class Tokenizer:

    SOT = "<|sot|>" # start of title
    EOT = "<|eot|>" # end of title
    SOA = "<|soa|>" # start of article
    EOA = "<|eoa|>" # end of article
    UNK = "<|unk|>" # unknown token
    PAD = "<|pad|>" # padding token
    special_tokens_ = [SOT, EOT, SOA, EOA, UNK, PAD]

    # regex pattern for splitting the text
    pattern_ = ".".join(special_tokens_).replace("|", "\|").replace(".", "|") + \
            "|'t|'s|'re|'ve|'d|'ll|'m|'em| ?[A-Za-z]+| ?[a-z]+| ?[0-9]{1,4}| ?[^A-Za-z0-9\s]|\s+(?!\S)|\s+"
    splitter = re.compile(rf"{pattern_}")

    @classmethod
    def get_str_tokens(cls, text):
        """
        Tokenize the input text into a list of tokens.

        Args:
            text (str) : The input text.
        
        Returns:
            List[str] : The list of tokens.
        """
        return cls.splitter.findall(string=text)

    def __init__(self, vocab):
        """
        Tokenizer class for converting text to numbers and vice versa.

        Args:
            vocab (Dict[str, int]) : The vocabulary dictionary.
        """
        self.encode_vocab = {k: int(v) for k, v in vocab.items()}
        self.decode_vocab = {int(v): k for k, v in vocab.items()}

    def __len__(self):
        """
        Return the size of vocabulary.

        Returns:
            int: The size of vocabulary.
        """
        return len(self.encode_vocab)

    def encode(self, text):
        """
        Tokenize the input text into a list of tokens.

        Args:
            text (str) : The input text.
        
        Returns:
            List[str] : The list of tokens.
        """
        tokens = self.get_str_tokens(text=text)
        return list(map(lambda tk: self.encode_vocab.get(tk, self.encode_vocab.get(self.UNK)), tokens))

    def decode(self, indices):
        """
        Decode the list of indices into a text.

        Args:
            indices (List[int]) : The list of indices.
        
        Returns:
            str : The output text.
        """
        return "".join([self.decode_vocab.get(idx, self.UNK) for idx in indices])
