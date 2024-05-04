import asyncio
import os

import httpx
import streamlit as st

# env variables
IP_HOST = os.getenv(key="IP_HOST", default="0.0.0.0")
PORT = os.getenv(key="PORT", default=8000)

# main variables
URL = f"http://{IP_HOST}:{PORT}"


async def generate_text(text, max_tokens):
    """Request the Medium Article Generator API to generate text based on the prompt."""
    async with httpx.AsyncClient() as client:
        try:
            async with client.stream(
                method="GET",
                url=f"{URL}/generate",
                params={"text": text, "max_tokens": max_tokens}
            ) as response:
                async for chunk in response.aiter_text():
                    yield chunk
        except Exception as error:
            yield "<|API-ERROR|>"


async def main():
    """Run this function to display the Medium Article Generator app."""
    st.title(body="Medium Article Generator")

    # text generation field
    prompt = st.text_input(
        label="Prompt",
        placeholder="Initial text to generate the article",
        max_chars=50
    ).strip()
    max_tokens = st.number_input(label="Tokens", min_value=50, step=1, help="Total number of tokens to be generated.")

    output = st.empty()
    if prompt:
        with st.spinner(text="Generating text..."):
            full_text = "" # store the full generation
            async for text in generate_text(text=prompt, max_tokens=max_tokens):
                if "<|API-ERROR|>" in text:
                    st.error(body="Error while generating text.")
                    break
                full_text += text
                output.write(full_text)
        st.divider()

if __name__=="__main__":
    asyncio.run(main())
