# reward_machines

This repository contains a script to generate noise plots for various algorithms. It reads data from predefined files, processes the data to compute the median, 25th percentile, and 75th percentile, and then plots the results. This code is an extended and modernized version of the original reward machine system designed by Icarte et al. (https://github.com/RodrigoToroIcarte/reward_machines).

## Dependencies

To run the script, you need the following dependencies:

- Python 3.11
- stable-baselines3
- gym (not gymnasium)
- pandas
- matplotlib
- numpy
- etc.

## HOW TO RUN
The scripts folder contains almost all of the commands that were run for the experiment. An example of a command (for CRM) that you can run without running scripts because they are taking time is: 
- python3 run.py --alg=qlearning --env=Mordor-single-v0 --num_timesteps=1e5 --gamma=0.9 --use_crm
ONE NOTE: monitor.py in the stable-baselines3 library (the error log says the exact location) needs to be changed for qlearning scripts, namely line 94 should have only four arguments (without truncated). Furthermore, for all scripts, truncated in the monitor.py file needs to be always set to false.

To adjust the noise level, go to the file rm_environment.py in reward_machines folder and change it in line 89.
