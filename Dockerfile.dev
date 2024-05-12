# base image
FROM --platform=amd64 ubuntu:22.04
# install dependencies
RUN apt-get update && apt-get install -y curl wget python3 python3-pip
# working directory
WORKDIR /home
# copy requirements file
COPY model/requirements.txt model_requirements.txt
COPY dev/requirements.txt dev_requirements.txt
# install requirements
RUN pip install -r model_requirements.txt
RUN pip install -r dev_requirements.txt
# copy project
COPY . .