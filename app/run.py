"""
Main module that runs the API of Medium Article Generator model.
"""
import os
import sys

abs_path = os.path.abspath(path=os.path.dirname(p=__file__))
abs_path = os.path.join(os.sep, *abs_path)
sys.path.append(abs_path)

from fastapi import FastAPI, status
from fastapi.responses import RedirectResponse, Response

from app.schemas import ModelDetails, SourceData

app = FastAPI()


@app.get(path="/", status_code=status.HTTP_308_PERMANENT_REDIRECT)
async def model_page():
    """Redirects to the Medium article of model."""
    return RedirectResponse(url="/", status_code=status.HTTP_308_PERMANENT_REDIRECT)

@app.get(path="/about", status_code=status.HTTP_200_OK)
async def model_details():
    """Returns all hyper-parameters of model."""
    return ModelDetails(
        num_layers=6,
        embedding_dim=768,
        head_dim=64,
        feed_forward_dim=3072,
        context_window=512
    )

@app.get(path="/download/{source}", status_code=status.HTTP_307_TEMPORARY_REDIRECT)
async def get_model_weights(source: str, version: str = "1.0"):
    """Download the pretrained weights of model."""
    try:
        file = SourceData[source].value
    except KeyError:
        return Response(content=f"Invalid source", status_code=status.HTTP_400_BAD_REQUEST)
    url = f"https://github.com/MTxSouza/MediumArticleGenerator/releases/download/{source}-v{version}/{file}"
    return RedirectResponse(url=url)
