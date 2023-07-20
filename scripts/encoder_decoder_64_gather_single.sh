#!/bin/bash

dataset=$1
seed=$2
gpu=$3

echo "===== Gathering encoder-decoder-64 model on: $dataset (gpu id: $gpu) ====="

screen -S gather-"$dataset"-64 -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model64/config"$seed".yaml ./logs_"$dataset"_encoder-decoder-64_"$seed"/lightning_logs/checkpoints NA eval-train; \
                                          exec sh";