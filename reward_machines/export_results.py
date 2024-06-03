"""
This code exports summary results for all our approaches and runs.
"""

import math,os,statistics,argparse
from collections import deque
import numpy as np

def get_precentiles_str(a):
    p25 = "%0.4f"%float(np.percentile(a, 25))
    p50 = "%0.4f"%float(np.percentile(a, 50))
    p75 = "%0.4f"%float(np.percentile(a, 75))
    return [p25, p50, p75]

def export_avg_results_noise(agent,env,maps,seeds):
    """
    NOTE: 
        - Find a way to summarize the results coming from different seeds
        - This is tricky because the timesteps per trace might be different
    """

    if 'office' in env:
        num_episodes_avg = 400
        num_total_steps = 1e5
        max_length = 100 
        # These values were computed using python3 test_optimal_policies.py --env Office-v0
        optimal_rewards = dict(M1=[0.07293349939375451, 0.031371219723777224, 0.031344625839253674, 0.028068145364196])
    elif 'craft' in env:
        num_episodes_avg = 1000
        num_total_steps = 2e6
        max_length = 200 
        # These values were computed using python3 test_optimal_policies.py --env Craft-Mx-v0, where Mx \in {M0,...,M10}
        optimal_rewards = dict(
            M0=[0.021056896352971022, 0.02223704553392025, 0.030381492031773438, 0.029568570099147824, 0.028591107021728984, 0.01698443909619106, 0.01846431419356979, 0.022195745620423268, 0.023432205498423144, 0.012990580561337057],
            M1=[0.029351862026684234, 0.08145408101872667, 0.03686954558213452, 0.04016888761057505, 0.02377734567512407, 0.016688913949401675, 0.019328533833491517, 0.026399431925736412, 0.014480703440661439, 0.01608448909869953],
            M2=[0.02746240268841621, 0.03393122229725049, 0.04673408609871168, 0.029393211012818346, 0.020403425533077623, 0.02156600149255032, 0.019132586719891882, 0.025771297783600298, 0.012038910965727262, 0.016267593882624505],
            M3=[0.040880112777651245, 0.02627265769813315, 0.03260017413152539, 0.03037111900115712, 0.02386404446283515, 0.01897572813234741, 0.01923301504368416, 0.02614185297297031, 0.01281291791627108, 0.015542177173904392],
            M4=[0.049983622856357844, 0.032755395903900385, 0.0464494618843689, 0.06233224590891654, 0.027226834352011717, 0.02688353890797752, 0.018288920648559186, 0.027138286487400933, 0.016647263900480186, 0.015153066220061151],
            M5=[0.04206598764447244, 0.035589139246599184, 0.09991207351414957, 0.04632571635000513, 0.03702528830677027, 0.02398442705841878, 0.018728769938167387, 0.020487412797804926, 0.03412880060515614, 0.01425830298231041],
            M6=[0.03177125207281578, 0.038210879166676474, 0.022897777929546902, 0.031786611176422425, 0.021664205327344456, 0.014502414931619873, 0.018775529129346154, 0.02393232492474861, 0.015193935781772498, 0.017646578127461003],
            M7=[0.03840135248002624, 0.03718683914731578, 0.026967022534129442, 0.023968655442161438, 0.026031991060988944, 0.023128298094237006, 0.015929041669999186, 0.02595143142479329, 0.020006350750453, 0.01436104192390331],
            M8=[0.03524655946346928, 0.022589615162941463, 0.037256790344040905, 0.028548915407926288, 0.015094513551064151, 0.01758108253017802, 0.014691452663558889, 0.021827438920964642, 0.012412559851792787, 0.010353648029998358],
            M9=[0.04065716304804992, 0.028279235440572967, 0.03787076165651398, 0.0497159731275612, 0.02381682895885013, 0.02049120604804244, 0.016923128041790636, 0.028248394849629, 0.017381356260630407, 0.01058547554394297],
            M10=[0.027431983274420882, 0.06260777145288554, 0.034241716927835716, 0.05593015110915622, 0.02474730823989501, 0.0187448744296349, 0.019920828933960858, 0.03675192807314101, 0.014629262051232932, 0.015968906266383056]
            )        
    else:
        assert False, "Invalid environment!"


    #Loop over each noise model
    for noise in range(10, 99, 10):
        stats = [[] for _ in range(max_length)]
        # Loop over each map of an environment
        for env_map in maps:
            # Loop 60 times
            for seed in seeds:
                # Reading the results -> format: list of [time, episode length, episode reward]
                f_path = "../noise_results/%s/%s/%s/%s/%s/0.0.monitor.csv"%(noise, agent, env, env_map, seed)
                results = []
                f = open(f_path)
                for l in f:
                    raw = l.strip().split(',')
                    if len(raw) != 3 or raw[0]=='r':
                        continue
                    episode_reward, episode_length, t = float(raw[0]), float(raw[1]), float(raw[2])
                    results.append((t, episode_length, episode_reward))
                f.close()

                # collecting average stats
                steps = 0
                rewards = deque([], maxlen=num_episodes_avg)
                steps_tic = num_total_steps / max_length
                for i in range(len(results)):
                    _, episode_length, episode_reward = results[i]
                    rew_per_step = (episode_reward / episode_length) / optimal_rewards[env_map][i % len(optimal_rewards[env_map])]
                    if (steps + episode_length) % steps_tic == 0:
                        steps += episode_length
                        rewards.append(rew_per_step)
                        stats[int((steps + episode_length) // steps_tic) - 1].append(sum(rewards) / len(rewards))
                    else:
                        if (steps // steps_tic) != (steps + episode_length) // steps_tic:
                            stat_id = int((steps + episode_length) // steps_tic) - 1
                            # For some reason it would go over 100 by one
                            if stat_id == 100:
                                #stat_id -= 1
                                continue
                            stats[stat_id].append(sum(rewards) / len(rewards))
                        steps += episode_length
                        rewards.append(rew_per_step)
                    if (steps + episode_length) // steps_tic == max_length:
                        break

        # Saving the average performance and standard deviation
        f_out = "../noise_results/%s/summary/%s-%s.txt"%(noise, env, agent)
        f = open(f_out, 'w')
        for i in range(max_length):
            if len(stats[i]) == len(seeds) * len(maps):
                f.write("\t".join([str((i + 1) * steps_tic / 1000)] + get_precentiles_str(stats[i])) + "\n")
        f.close()

def export_avg_results_noise_single(agent,env,maps,seeds):
    """
    NOTE: 
        - Find a way to summarize the results coming from different seeds
        - This is tricky because the timesteps per trace might be different
    """

    if 'office' in env:
        num_episodes_avg = 100
        num_total_steps = 1e5
        max_length = 100 
        # These values were computed using python3 test_optimal_policies.py --env Office-v0
        optimal_rewards = dict(M1=0.031344625839253674)
    elif 'craft' in env:
        num_episodes_avg = 100
        num_total_steps = 2e6
        max_length = 200 
        # These values were computed using python3 test_optimal_policies.py --env Craft-Mx-v0, where Mx \in {M0,...,M10}
        optimal_rewards = dict(M0=0.012990580561337057, M1=0.01608448909869953, M2=0.016267593882624505, M3=0.015542177173904392, M4=0.015153066220061151, M5=0.01425830298231041, M6=0.017646578127461003, M7=0.01436104192390331, M8=0.010353648029998358, M9=0.01058547554394297, M10=0.015968906266383056)        
    else:
        assert False, "Invalid environment!"

    for noise in range(10, 99, 10):
        stats = [[] for _ in range(max_length)]
        for env_map in maps:
            for seed in seeds:
                # Reading the results
                f_path = "../noise_results/%s/%s/%s/%s/%s/0.0.monitor.csv"%(noise, agent, env, env_map, seed)
                results = []
                f = open(f_path)
                for l in f:
                    raw = l.strip().split(',')
                    if len(raw) != 3 or raw[0]=='r':
                        continue
                    r,l,t = float(raw[0]), float(raw[1]), float(raw[2])
                    results.append((t,l,r))
                f.close()

                # collecting average stats
                steps = 0
                rewards = deque([], maxlen=num_episodes_avg)
                steps_tic = num_total_steps/max_length
                for i in range(len(results)):
                    _,l,r = results[i]
                    rew_per_step = (r/l)/optimal_rewards[env_map]
                    if (steps+l)%steps_tic == 0:
                        steps += l
                        rewards.append(rew_per_step)
                        stats[int((steps+l)//steps_tic)-1].append(sum(rewards)/len(rewards))
                    else:
                        if (steps//steps_tic) != (steps+l)//steps_tic:
                            stats[int((steps+l)//steps_tic)-1].append(sum(rewards)/len(rewards))
                        steps += l
                        rewards.append(rew_per_step)
                    if (steps+l)//steps_tic == max_length:
                        break

        # Saving the average performance and standard deviation
        f_out = "../noise_results/%s/summary/%s-%s.txt"%(noise, env, agent)
        f = open(f_out, 'w')
        for i in range(max_length):
            if len(stats[i]) == len(seeds) * len(maps):
                #f.write("\t".join([str((i+1)*steps_tic/1000), "%0.4f"%(sum(stats[i])/len(stats[i]))]) + "\n")
                f.write("\t".join([str((i+1)*steps_tic/1000)] + get_precentiles_str(stats[i])) + "\n")
        f.close()

def export_avg_results_grid(agent,env,maps,seeds):
    """
    NOTE: 
        - Find a way to summarize the results coming from different seeds
        - This is tricky because the timesteps per trace might be different
    """

    if 'office' in env:
        num_episodes_avg = 400
        num_total_steps = 1e5
        max_length = 100 
        # These values were computed using python3 test_optimal_policies.py --env Office-v0
        optimal_rewards = dict(M1=[0.07293349939375451, 0.031371219723777224, 0.031344625839253674, 0.028068145364196])
    elif 'craft' in env:
        num_episodes_avg = 1000
        num_total_steps = 2e6
        max_length = 200 
        # These values were computed using python3 test_optimal_policies.py --env Craft-Mx-v0, where Mx \in {M0,...,M10}
        optimal_rewards = dict(
            M0=[0.021056896352971022, 0.02223704553392025, 0.030381492031773438, 0.029568570099147824, 0.028591107021728984, 0.01698443909619106, 0.01846431419356979, 0.022195745620423268, 0.023432205498423144, 0.012990580561337057],
            M1=[0.029351862026684234, 0.08145408101872667, 0.03686954558213452, 0.04016888761057505, 0.02377734567512407, 0.016688913949401675, 0.019328533833491517, 0.026399431925736412, 0.014480703440661439, 0.01608448909869953],
            M2=[0.02746240268841621, 0.03393122229725049, 0.04673408609871168, 0.029393211012818346, 0.020403425533077623, 0.02156600149255032, 0.019132586719891882, 0.025771297783600298, 0.012038910965727262, 0.016267593882624505],
            M3=[0.040880112777651245, 0.02627265769813315, 0.03260017413152539, 0.03037111900115712, 0.02386404446283515, 0.01897572813234741, 0.01923301504368416, 0.02614185297297031, 0.01281291791627108, 0.015542177173904392],
            M4=[0.049983622856357844, 0.032755395903900385, 0.0464494618843689, 0.06233224590891654, 0.027226834352011717, 0.02688353890797752, 0.018288920648559186, 0.027138286487400933, 0.016647263900480186, 0.015153066220061151],
            M5=[0.04206598764447244, 0.035589139246599184, 0.09991207351414957, 0.04632571635000513, 0.03702528830677027, 0.02398442705841878, 0.018728769938167387, 0.020487412797804926, 0.03412880060515614, 0.01425830298231041],
            M6=[0.03177125207281578, 0.038210879166676474, 0.022897777929546902, 0.031786611176422425, 0.021664205327344456, 0.014502414931619873, 0.018775529129346154, 0.02393232492474861, 0.015193935781772498, 0.017646578127461003],
            M7=[0.03840135248002624, 0.03718683914731578, 0.026967022534129442, 0.023968655442161438, 0.026031991060988944, 0.023128298094237006, 0.015929041669999186, 0.02595143142479329, 0.020006350750453, 0.01436104192390331],
            M8=[0.03524655946346928, 0.022589615162941463, 0.037256790344040905, 0.028548915407926288, 0.015094513551064151, 0.01758108253017802, 0.014691452663558889, 0.021827438920964642, 0.012412559851792787, 0.010353648029998358],
            M9=[0.04065716304804992, 0.028279235440572967, 0.03787076165651398, 0.0497159731275612, 0.02381682895885013, 0.02049120604804244, 0.016923128041790636, 0.028248394849629, 0.017381356260630407, 0.01058547554394297],
            M10=[0.027431983274420882, 0.06260777145288554, 0.034241716927835716, 0.05593015110915622, 0.02474730823989501, 0.0187448744296349, 0.019920828933960858, 0.03675192807314101, 0.014629262051232932, 0.015968906266383056]
            )        
    else:
        assert False, "Invalid environment!"


    stats = [[] for _ in range(max_length)]
    for env_map in maps:
        for seed in seeds:
            # Reading the results
            f_path = "../negative_reward_results/%s/%s/%s/%s/0.0.monitor.csv"%(agent,env,env_map,seed)
            results = []
            f = open(f_path)
            for l in f:
                raw = l.strip().split(',')
                if len(raw) != 3 or raw[0]=='r':
                    continue
                r,l,t = float(raw[0]), float(raw[1]), float(raw[2])
                results.append((t,l,r))
            f.close()

            # collecting average stats
            steps = 0
            rewards = deque([], maxlen=num_episodes_avg)
            steps_tic = num_total_steps/max_length
            for i in range(len(results)):
                _,l,r = results[i]
                rew_per_step = (r/l)/optimal_rewards[env_map][i%len(optimal_rewards[env_map])]
                if (steps+l)%steps_tic == 0:
                    steps += l
                    rewards.append(rew_per_step)
                    stats[int((steps+l)//steps_tic)-1].append(sum(rewards)/len(rewards))
                else:
                    if (steps//steps_tic) != (steps+l)//steps_tic:    
                        stats[int((steps+l)//steps_tic)-1].append(sum(rewards)/len(rewards))
                    steps += l
                    rewards.append(rew_per_step)
                if (steps+l)//steps_tic == max_length:
                    break

    # Saving the average performance and standard deviation
    f_out = "../negative_reward_results/summary/%s-%s.txt"%(env,agent)
    f = open(f_out, 'w')
    for i in range(max_length):
        if len(stats[i]) == len(seeds) * len(maps):
            #f.write("\t".join([str((i+1)*steps_tic/1000), "%0.4f"%(sum(stats[i])/len(stats[i]))]) + "\n")
            f.write("\t".join([str((i+1)*steps_tic/1000)] + get_precentiles_str(stats[i])) + "\n")
    f.close()

def export_avg_results_grid_single(agent,env,maps,seeds):
    """
    NOTE: 
        - Find a way to summarize the results coming from different seeds
        - This is tricky because the timesteps per trace might be different
    """

    if 'office' in env:
        num_episodes_avg = 100
        num_total_steps = 1e5
        max_length = 100 
        # These values were computed using python3 test_optimal_policies.py --env Office-v0
        optimal_rewards = dict(M1=0.031344625839253674)
        
    elif 'craft' in env:
        num_episodes_avg = 100
        num_total_steps = 2e6
        max_length = 200 
        # These values were computed using python3 test_optimal_policies.py --env Craft-Mx-v0, where Mx \in {M0,...,M10}
        optimal_rewards = dict(M0=0.012990580561337057, M1=0.01608448909869953, M2=0.016267593882624505, M3=0.015542177173904392, M4=0.015153066220061151, M5=0.01425830298231041, M6=0.017646578127461003, M7=0.01436104192390331, M8=0.010353648029998358, M9=0.01058547554394297, M10=0.015968906266383056)        
    else:
        assert False, "Invalid environment!"


    stats = [[] for _ in range(max_length)]
    for env_map in maps:
        for seed in seeds:
            # Reading the results
            f_path = "../negative_reward_results/%s/%s/%s/%s/0.0.monitor.csv"%(agent,env,env_map,seed)
            results = []
            f = open(f_path)
            for l in f:
                raw = l.strip().split(',')
                if len(raw) != 3 or raw[0]=='r':
                    continue
                r,l,t = float(raw[0]), float(raw[1]), float(raw[2])
                results.append((t,l,r))
            f.close()

            # collecting average stats
            steps = 0
            rewards = deque([], maxlen=num_episodes_avg)
            steps_tic = num_total_steps/max_length
            for i in range(len(results)):
                _,l,r = results[i]
                rew_per_step = (r/l)/optimal_rewards[env_map]
                if (steps+l)%steps_tic == 0:
                    steps += l
                    rewards.append(rew_per_step)
                    stats[int((steps+l)//steps_tic)-1].append(sum(rewards)/len(rewards))
                else:
                    if (steps//steps_tic) != (steps+l)//steps_tic:
                        stats[int((steps+l)//steps_tic)-1].append(sum(rewards)/len(rewards))
                    steps += l
                    rewards.append(rew_per_step)
                if (steps+l)//steps_tic == max_length:
                    break

    # Saving the average performance and standard deviation
    f_out = "../negative_reward_results/summary/%s-%s.txt"%(env,agent)
    f = open(f_out, 'w')
    for i in range(max_length):
        if len(stats[i]) == len(seeds) * len(maps):
            #f.write("\t".join([str((i+1)*steps_tic/1000), "%0.4f"%(sum(stats[i])/len(stats[i]))]) + "\n")
            f.write("\t".join([str((i+1)*steps_tic/1000)] + get_precentiles_str(stats[i])) + "\n")
    f.close()

if __name__ == '__main__':
    algs = ['ql', 'crm', 'hrm', 'hrm-rs', 'ql-rs','crm-rs']

    # Office world (multitask)
    for alg in algs:
        print(alg,'office')
        export_avg_results_grid(alg,'office',['M1'],list(range(60)))

    # Office world (single task)
    for alg in algs:
        print(alg,'office-single')
        export_avg_results_grid_single(alg,'office-single',['M1'],list(range(60)))
