#!/bin/bash

dataset=$1
seed=$2

echo "===== Evaluating symbolic regression model on: $dataset ====="

screen -S eval-"$dataset"-sr_"$seed" -dm bash -c "python ../eval.py ../configs/"$dataset"/regression/config"$seed".yaml ./logs_"$dataset"_refine-64_"$seed"/symbolic_regression/model.pkl NA eval-sr-train; \
                                                  exec sh";