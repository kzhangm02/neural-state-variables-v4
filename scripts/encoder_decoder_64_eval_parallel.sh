#!/bin/bash

dataset=$1
gpu1=$2
gpu2=$3
gpu3=$4

echo "===== Evaluating encoder-decoder-64 model on: $dataset (gpu ids: $gpu1, $gpu2, $gpu3) ====="

screen -S eval-"$dataset"-64_1 -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu1" python ../eval.py ../configs/"$dataset"/model64/config1.yaml ../logs/"$dataset"_encoder-decoder-64_1/lightning_logs/checkpoints NA eval-eval; \
                                             exec sh";
screen -S eval-"$dataset"-64_2 -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu2" python ../eval.py ../configs/"$dataset"/model64/config2.yaml ../logs/"$dataset"_encoder-decoder-64_2/lightning_logs/checkpoints NA eval-eval; \
                                             exec sh";
screen -S eval-"$dataset"-64_3 -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu3" python ../eval.py ../configs/"$dataset"/model64/config3.yaml ../logs/"$dataset"_encoder-decoder-64_3/lightning_logs/checkpoints NA eval-eval; \
                                             exec sh";