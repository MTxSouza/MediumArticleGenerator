import re

from app.schema import Prompt


def check_prompt(prompt: Prompt):
    """
    Check if the prompt text is valid.
    
    Args:
        prompt (Prompt) : The prompt object.
    
    Returns:
        tuple : The input text, number of extra tokens, and maximum length.
    """
    # Retrieving parameters
    text = prompt.text
    extra_tokens = prompt.extra_tokens
    max_len = prompt.max_length

    # Converting text to lowercase
    text = text.lower()

    # Checking parameters
    _text = re.sub(pattern=r"\s+", repl=" ", string=text).strip() # Removing extra spaces
    if not _text:
        raise ValueError("The input text should not be empty.")
    elif re.match(pattern=r"[^\w ]+", string=_text):
        raise ValueError("The input text should contain only letters, digits, and spaces.")
    elif not re.match(pattern=r"[\w ]{8,30}", string=_text):
        raise ValueError("The input text should contain at least 8 characters and at most 30.")

    if extra_tokens < 50 or extra_tokens > 100:
        raise ValueError("The number of extra tokens should be between 50 and 100.")
    
    if max_len is not None and max_len < 80:
        raise ValueError("The maximum length should be at least 80 tokens.")
    max_len = float("inf") if max_len is None else max_len

    return _text, extra_tokens, max_len
