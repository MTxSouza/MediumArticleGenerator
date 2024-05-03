#!/bin/bash
wget https://github.com/MTxSouza/MediumArticleGenerator/releases/download/model-v1.0/weights.pt
mkdir -p source
mv weights.pt source/weights.pt