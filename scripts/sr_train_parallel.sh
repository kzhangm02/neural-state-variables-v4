#!/bin/bash

dataset=$1

echo "===== Training symbolic regression model on: $dataset ====="

screen -S train-"$dataset"-sr_1 -dm bash -c "python ../sr.py ../configs/"$dataset"/regression/config1.yaml; \
                                             exec sh";
screen -S train-"$dataset"-sr_2 -dm bash -c "python ../sr.py ../configs/"$dataset"/regression/config2.yaml; \
                                             exec sh";
screen -S train-"$dataset"-sr_3 -dm bash -c "python ../sr.py ../configs/"$dataset"/regression/config3.yaml; \
                                             exec sh";