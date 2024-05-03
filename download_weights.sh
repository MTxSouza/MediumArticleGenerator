#!/bin/bash
wget https://github.com/MTxSouza/MediumArticleGenerator/releases/download/model-v1.0/model.zip
mkdir -p source
unzip model.zip -d source
rm model.zip