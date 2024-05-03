"""
Base module that defines the main logger object of API.
"""
import logging
import os

# absolute path for logs/
logs_folder = os.path.abspath(path=os.path.dirname(p=__file__))
logs_folder = os.path.join(logs_folder, "logs")
if not os.path.exists(path=logs_folder):
    os.mkdir(path=logs_folder)

# formatter
fmt = logging.Formatter(fmt="[%(levelname)-8s]: %(asctime)s - %(message)s ; module `%(module)s` in line %(lineno)s")

# file
hdlr = logging.FileHandler(filename=os.path.join(logs_folder, "system.log"), mode="w")
hdlr.setFormatter(fmt=fmt)

# logger
logger = logging.getLogger(name="pose-detection-logger")
logger.addHandler(hdlr=hdlr)
logger.setLevel(level=logging.DEBUG)

# default messages
INTERNAL_ERROR_MSG = "Internal error. Checks `system.log` for more details located in app/logs/."
