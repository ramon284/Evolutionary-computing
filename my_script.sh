#!/bin/bash

ctr=1
enemy_type=3
while [ $ctr -le 10 ]
do
    python3 elitism.py --run_no $ctr --exp_name elitism_type_$enemy_type --enemy_type $enemy_type
    ((ctr++))
done
