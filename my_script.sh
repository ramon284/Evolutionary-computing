#!/bin/bash

ctr=1
enemy_type=256
while [ $ctr -le 10 ]
do
    python3 neat-evoman.py --run_no $ctr --enemy_type $enemy_type
    ((ctr++))
done
