#!/bin/bash

dataset=$1

echo "===== Evaluating symbolic regression model on: $dataset ====="

screen -S eval-"$dataset"-sr_1 -dm bash -c "python ../eval.py ../configs/"$dataset"/regression/config1.yaml ./logs_"$dataset"_refine-64_1/symbolic_regression/model.pkl NA eval-sr-train; \
                                            exec sh";
screen -S eval-"$dataset"-sr_2 -dm bash -c "python ../eval.py ../configs/"$dataset"/regression/config2.yaml ./logs_"$dataset"_refine-64_2/symbolic_regression/model.pkl NA eval-sr-train; \
                                            exec sh";
screen -S eval-"$dataset"-sr_3 -dm bash -c "python ../eval.py ../configs/"$dataset"/regression/config3.yaml ./logs_"$dataset"_refine-64_3/symbolic_regression/model.pkl NA eval-sr-train; \
                                            exec sh";