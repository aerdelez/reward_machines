#!/bin/bash
cd ../reward_machines
for i in `seq 0 59`; 
do
	# Multi-task
	python3 run.py --alg=qlearning --env=Mordor-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../mordor_results/ql-rs/mordor/M1/$i --use_rs
	python3 run.py --alg=qlearning --env=Mordor-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../mordor_results/crm/mordor/M1/$i --use_crm
	python3 run.py --alg=qlearning --env=Mordor-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../mordor_results/crm-rs/mordor/M1/$i --use_crm --use_rs
	python3 run.py --alg=hrm --env=Mordor-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../mordor_results/10/hrm/mordor/M1/$i
	python3 run.py --alg=hrm --env=Mordor-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../mordor_results/10/hrm-rs/mordor/M1/$i --use_rs

	# Single task
	python3 run.py --alg=qlearning --env=Mordor-single-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../mordor_results/ql-rs/mordor-single/M1/$i --use_rs
	python3 run.py --alg=qlearning --env=Mordor-single-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../mordor_results/crm/mordor-single/M1/$i --use_crm
	python3 run.py --alg=qlearning --env=Mordor-single-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../mordor_results/crm-rs/mordor-single/M1/$i --use_crm --use_rs
	python3 run.py --alg=hrm --env=Mordor-single-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../mordor_results/hrm/mordor-single/M1/$i
	python3 run.py --alg=hrm --env=Mordor-single-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../mordor_results/hrm-rs/mordor-single/M1/$i --use_rs
done