#!/usr/bin/env bash
python3 utils/baseline_centralized.py --num_parallel_tree 1 
python3 utils/baseline_centralized.py --num_parallel_tree 5 --subsample 0.8
python3 utils/baseline_centralized.py --num_parallel_tree 5 --subsample 0.2
python3 utils/baseline_centralized.py --num_parallel_tree 20 --subsample 0.05
python3 utils/baseline_centralized.py --num_parallel_tree 20 --subsample 0.8


