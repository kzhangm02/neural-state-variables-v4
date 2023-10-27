#!/bin/bash

dataset=$1
seed=$2
gpu=$3

echo "===== Training encoder-decoder-64 model on: $dataset (gpu ids: $gpu) ====="

screen -S train-"$dataset"-64 -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu" python ../main.py ../configs/"$dataset"/model64/config"$seed".yaml NA; \
                                           exec sh";