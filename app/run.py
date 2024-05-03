"""
Main module that runs the API of Medium Article Generator model.
"""
import os
import sys

abs_path = os.path.abspath(path=os.path.dirname(p=__file__))
abs_path = os.path.join(os.sep, *abs_path)
sys.path.append(abs_path)

from fastapi import FastAPI, status
from fastapi.responses import RedirectResponse

from app.config import params

app = FastAPI()


@app.get(path="/", status_code=status.HTTP_308_PERMANENT_REDIRECT)
async def model_page():
    """Redirects to the Medium article of model."""
    return RedirectResponse(url="/", status_code=status.HTTP_308_PERMANENT_REDIRECT)

@app.get(path="/about", status_code=status.HTTP_200_OK)
async def model_details():
    """Returns all hyper-parameters of model."""
    return params
