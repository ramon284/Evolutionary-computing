#!/bin/bash

ctr=1
enemy_type=2
while [ $ctr -le 10 ]
do
    python3 elitism_2.py --run_no $ctr --exp_name elitism --enemy_type $enemy_type --run_mode train
    python3 elitism_2.py --run_no $ctr --exp_name elitism --enemy_type $enemy_type --run_mode test
    python3 elitism_2.py --run_no $ctr --exp_name elitism --enemy_type $enemy_type --run_mode test
    python3 elitism_2.py --run_no $ctr --exp_name elitism --enemy_type $enemy_type --run_mode test
    python3 elitism_2.py --run_no $ctr --exp_name elitism --enemy_type $enemy_type --run_mode test
    python3 elitism_2.py --run_no $ctr --exp_name elitism --enemy_type $enemy_type --run_mode test
    ((ctr++))
done
