#!/bin/bash
cd ../reward_machines
for i in `seq 0 59`; 
do
	# Multi-task
	python3 run.py --alg=qlearning --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/ql/office/M1/$i

	# Single task
	python3 run.py --alg=qlearning --env=Office-single-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/ql/office-single/M1/$i
done