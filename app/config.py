"""
Initialize the API and instanciate the pre-trained model for use.
"""
import json
import os
import sys

import torch

from app.logger import INTERNAL_ERROR_MSG, logger
from model import ArticleGenerator
from model.tokenizer import Tokenizer

logger.info(msg="Initializing API.")

# absolute path of project
absolute_path = os.path.abspath(path=os.path.dirname(p=__file__))
absolute_path = os.path.join(os.sep, *absolute_path.split(sep=os.sep)[:-1])
logger.debug(msg=f"Absolute folder path of project : {absolute_path}.")

# model
logger.info(msg="Loading model..")

source_folder_path = os.path.join(absolute_path, "source")
if not os.path.exists(path=source_folder_path):
    logger.error(msg="API could not find the `source` directory in project root folder.")
    sys.exit()

weights_filepath = os.path.join(source_folder_path, "weights.pt")
logger.debug(msg=f"Absolute path of weights file : {weights_filepath}.")
vocab_filepath = os.path.join(source_folder_path, "vocab.json")
logger.debug(msg=f"Absolute path of params file : {vocab_filepath}.")
mapper_filepath = os.path.join(source_folder_path, "mapper.json")
logger.debug(msg=f"Absolute path of params file : {mapper_filepath}.")
params_filepath = os.path.join(source_folder_path, "params.json")
logger.debug(msg=f"Absolute path of params file : {params_filepath}.")

for filepath in (weights_filepath, vocab_filepath, mapper_filepath, params_filepath):
    if not os.path.exists(path=filepath):
        filename = filepath.split(sep=os.sep)[-1]
        logger.error(msg=f"API could not find the `{filename}` file in `source` directory.")
        sys.exit()

device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
logger.debug(msg=f"Pytorch device : {device}")
if device.type == "cpu":
    logger.warning(msg="Pytorch will run the model on CPU, it's recommended to run on GPU for faster inference.")

# ===================== Tokenizer =====================
try:
    with open(file=vocab_filepath, mode="r", encoding="utf-8") as json_buffer:
        vocab = json.load(fp=json_buffer)
except Exception as error:
    print(INTERNAL_ERROR_MSG)
    logger.critical(msg=str(error))
    sys.exit()

try:
    with open(file=mapper_filepath, mode="r", encoding="utf-8") as json_buffer:
        mapper = json.load(fp=json_buffer)
except Exception as error:
    print(INTERNAL_ERROR_MSG)
    logger.critical(msg=str(error))
    sys.exit()

try:
    tokenizer = Tokenizer(vocab=vocab, lookup_vocab=mapper)
except Exception as error:
    print(INTERNAL_ERROR_MSG)
    logger.critical(msg=str(error))
    sys.exit()

# ===================== Model =====================
try:
    with open(file=params_filepath, mode="r", encoding="utf-8") as json_buffer:
        params = json.load(fp=json_buffer)
except Exception as error:
    print(INTERNAL_ERROR_MSG)
    logger.critical(msg=str(error))
    sys.exit()

try:
    model = ArticleGenerator(**params, vocab_size=len(tokenizer), device=device, tokenizer=tokenizer)
    model.to(device=device)
except Exception as error:
    print(INTERNAL_ERROR_MSG)
    logger.critical(msg=str(error))
    sys.exit()

# ===================== Weights =====================
try:
    weights = torch.load(f=weights_filepath, map_location=device)
except Exception as error:
    print(INTERNAL_ERROR_MSG)
    logger.critical(msg=str(error))
    sys.exit()

try:
    model.load_state_dict(state_dict=weights)
except Exception as error:
    print(INTERNAL_ERROR_MSG)
    logger.critical(msg=str(error))
    sys.exit()

logger.info(msg="API ready for use.")
