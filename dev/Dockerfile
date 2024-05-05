# base image
FROM --platform=amd64 ubuntu:22.04
RUN apt update -y && apt install -y python3 \
                                    python3-pip \
                                    wget

# repo
WORKDIR /home

# dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# files
COPY . .
CMD [ "nvidia-smi" ]