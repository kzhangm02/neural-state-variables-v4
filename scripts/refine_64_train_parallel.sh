#!/bin/bash

dataset=$1
gpu1=$2
gpu2=$3
gpu3=$4

echo "===== Training refine-64 model on: $dataset (gpu ids: $gpu1, $gpu2, $gpu3) ====="

screen -S train-"$dataset"-64_1 -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu1" python ../main.py ../configs/"$dataset"/refine64/config1.yaml; \
                                             exec sh";
screen -S train-"$dataset"-64_2 -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu2" python ../main.py ../configs/"$dataset"/refine64/config2.yaml; \
                                             exec sh";
screen -S train-"$dataset"-64_3 -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu3" python ../main.py ../configs/"$dataset"/refine64/config3.yaml; \
                                             exec sh";