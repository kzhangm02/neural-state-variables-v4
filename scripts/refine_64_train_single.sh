#!/bin/bash

dataset=$1
seed=$2
gpu=$3

echo "===== Training refine-64 model on: $dataset (gpu ids: $gpu) ====="

screen -S train-"$dataset"-64_"$seed" -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu" python ../main.py ../configs/"$dataset"/refine64/config"$seed".yaml ./logs_"$dataset"_encoder-decoder-64_"$seed"/lightning_logs/checkpoints; \
                                                   exec sh";