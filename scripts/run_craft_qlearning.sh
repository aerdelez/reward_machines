#!/bin/bash
cd ../reward_machines
for i in `seq 1 10`; 
do
	for j in `seq 0 5`; 
	do
		# Multi-task
		python3 run.py --alg=qlearning --env=Craft-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/ql/craft/M$i/$j

		# Single task
		python3 run.py --alg=qlearning --env=Craft-single-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/ql/craft-single/M$i/$j
	done
done