<img src="assets/Medium_Article_Generator.png" height="170px">

#### *Simple LLM trained over a dataset of Medium articles. Given a title to the model, it generates text related to that topic.*
[Medium Paper](https://medium.com/@msouza.os/llm-from-scratch-with-pytorch-9f21808c6319)

---

https://github.com/MTxSouza/MediumArticleGenerator/assets/67404745/b8cdfd30-edcd-4fad-a15f-413d07863701

## How to use
The model has an API for text generation, it recommended to run it inside a **Docker** container. The repo has a `docker-compose.yaml` file that build and initialize the API automatically. You can simply run the command below to build and start the API:
```bash
$ docker compose up api
```
When the API is running, open the `web/home.html` page on browers to access the UI to interact to the model!

---
