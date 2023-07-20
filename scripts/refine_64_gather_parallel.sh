#!/bin/bash

dataset=$1
gpu1=$2
gpu2=$3
gpu3=$4

echo "===== Gathering refine-64 model on: $dataset (gpu ids: $gpu1, $gpu2, $gpu3) ====="

screen -S gather-"$dataset"-64_1 -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu1" python ../eval.py ../configs/"$dataset"/refine64/config1.yaml ./logs_"$dataset"_encoder-decoder-64_1/lightning_logs/checkpoints ./logs_"$dataset"_refine-64_1/lightning_logs/checkpoints eval-refine-train; \
                                            exec sh";
screen -S gather-"$dataset"-64_2 -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu2" python ../eval.py ../configs/"$dataset"/refine64/config2.yaml ./logs_"$dataset"_encoder-decoder-64_2/lightning_logs/checkpoints ./logs_"$dataset"_refine-64_2/lightning_logs/checkpoints eval-refine-train; \
                                            exec sh";
screen -S gather-"$dataset"-64_3 -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu3" python ../eval.py ../configs/"$dataset"/refine64/config3.yaml ./logs_"$dataset"_encoder-decoder-64_3/lightning_logs/checkpoints ./logs_"$dataset"_refine-64_3/lightning_logs/checkpoints eval-refine-train; \
                                            exec sh";