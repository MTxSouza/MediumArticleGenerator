# Training

## About
This folder contains all necessary requirements to load the model and to train it.
##### Model
|Hyper-Parameter|Value|Details|
|-|-|-|
|Number of **Masked-Encoder** layers|6|-|
|Embedding dimmension|768|-|
|Head dimmension|64|**Embedding dimmension** / **Head dimmension** = Number of Heads|
|Context window|480|-|
|Number of FF neurons|3072|4 x **Embedding dimmension**|
|Dropout rate|5%|-|

##### Training
|Hyper-Parameter|Value|
|-|-|
|Epoch|1000|
|Batch-Size|12|
|Train iterations|400|
|Valid iterations|200|
|Learning Rate|2e-4|

##### Memory (recommended)
- VRAM >= 20Gb
- RAM >= 12Gb

*The memory above may be different if you try to train the model with differents hyper-parameters.*

## Docker
Run the command below to `build` and `run` the environment to train the model.
```
$ docker build -t image-caption-generator:train .
$ docker run -it --gpus all -v EXP_FOLDER:/home/notebook/exp/ image-caption-generator:train bash
```
details:
- EXP_FOLDER : Local directory on your computer to be mounted inside the container. During training, all generated files will be saved there and you can access them outside of container.
---