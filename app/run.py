"""
Main module that runs the API of Medium Article Generator model.
"""
from fastapi import FastAPI, status
from fastapi.exceptions import HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse

from app.config import logger, model, params
from app.schema import Prompt
from app.utils import check_prompt

app = FastAPI()


@app.get(path="/", status_code=status.HTTP_308_PERMANENT_REDIRECT)
async def model_page():
    """Redirects to the Medium article of model."""
    return RedirectResponse(url="/about", status_code=status.HTTP_308_PERMANENT_REDIRECT)

@app.get(path="/about", status_code=status.HTTP_200_OK)
async def model_details():
    """Returns all hyper-parameters of model."""
    return params

@app.post(path="/generate", status_code=status.HTTP_102_PROCESSING)
async def generate(prompt: Prompt):
    """Run the model inference and return it's generated text."""
    try:
        text, extra_tokens, max_length = check_prompt(prompt=prompt)
    except ValueError as error:
        logger.error(error.args[0])
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error.args[0]
        )
    except Exception as error:
        logger.critical(error.args[0])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error. Please try again later."
        )
    return StreamingResponse(
        content=model.generate(
            text=text,
            extra_tokens=extra_tokens,
            max_len=max_length
        ),
        status_code=status.HTTP_201_CREATED,
        media_type="text/plain"
    )