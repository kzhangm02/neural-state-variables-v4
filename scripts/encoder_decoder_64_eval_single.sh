#!/bin/bash

dataset=$1
seed=$2
gpu=$3

echo "===== Evaluating encoder-decoder-64 model on: $dataset (gpu ids: $gpu) ====="

screen -S eval-"$dataset"-64 -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model64/config"$seed".yaml ./logs_"$dataset"_encoder-decoder-64_"$seed"/lightning_logs/checkpoints NA eval-eval; \
                                          exec sh";