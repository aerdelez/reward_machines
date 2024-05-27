cd ../reward_machines
for i in `seq 0 59`; 
do
	# Multi-task
	python3 run.py --alg=qlearning --env=Mordor-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../mordor_noise_results/10/ql/mordor/M1/$i

	# Single task
	python3 run.py --alg=qlearning --env=Mordor-single-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../mordor_noise_results/10/ql/mordor-single/M1/$i
done