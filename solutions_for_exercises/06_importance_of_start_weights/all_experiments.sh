#!/bin/bash

echo "Started all experiments."

for seed_value in {02..02}
do
  python3 single_experiment.py $seed_value > logfile_$seed_value.txt
  #python3 single_experiment.py $seed_value
done

echo "All experiments finished!"