#!/bin/bash

echo "Started all experiments."

for seed_value in {01..03}
do
  python3 single_experiment.py $seed_value > logfile_$seed_value.txt
done

echo "All experiments finished!"