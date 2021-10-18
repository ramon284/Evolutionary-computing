#!/bin/bash

ctr=1
enemy_type=78
while [ $ctr -le 10 ]
do
    python3 elitism_and_distance.py --run_no $ctr --enemy_type $enemy_type --run_mode train
    python3 elitism_and_distance.py --run_no $ctr --enemy_type $enemy_type --run_mode test
    python3 elitism_and_distance.py --run_no $ctr --enemy_type $enemy_type --run_mode test
    python3 elitism_and_distance.py --run_no $ctr --enemy_type $enemy_type --run_mode test
    python3 elitism_and_distance.py --run_no $ctr --enemy_type $enemy_type --run_mode test
    python3 elitism_and_distance.py --run_no $ctr --enemy_type $enemy_type --run_mode test
    ((ctr++))
done
