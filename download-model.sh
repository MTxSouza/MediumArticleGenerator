#!/bin/bash
mkdir source -p
wget https://github.com/MTxSouza/MediumArticleGenerator/releases/download/model-v1.0/weights.pt
mv weights.pt source/weights.pt
wget https://github.com/MTxSouza/MediumArticleGenerator/releases/download/model-v1.0/params.json
mv params.json source/params.json