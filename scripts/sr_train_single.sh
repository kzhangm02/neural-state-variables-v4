#!/bin/bash

dataset=$1
seed=$2

echo "===== Training symbolic regression model on: $dataset ====="

screen -S train-"$dataset"-sr_"$seed" -dm bash -c "python ../sr.py ../configs/"$dataset"/regression/config"$seed".yaml; \
                                                   exec sh";