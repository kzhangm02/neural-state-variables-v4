#!/bin/bash

dataset=$1
seed=$2
gpu=$3

echo "===== Gathering refine-64 model on: $dataset (gpu id: $gpu) ====="

screen -S eval-"$dataset"-64 -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/refine64/config"$seed".yaml ../logs/"$dataset"_encoder-decoder-64_"$seed"/lightning_logs/checkpoints ../logs/"$dataset"_refine-64_"$seed"/lightning_logs/checkpoints eval-refine-train; \
                                          exec sh";