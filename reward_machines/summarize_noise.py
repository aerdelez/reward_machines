import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


environments = ['office', 'office-single', 'mordor', 'mordor-single']
algorithms = ['crm', 'crm-rs', 'hrm', 'hrm-rs', 'ql', 'ql-rs']

for environment in environments:
    main_df = pd.DataFrame(columns=algorithms)
    df_25th = pd.DataFrame(columns=algorithms)
    df_75th = pd.DataFrame(columns=algorithms)
    for algorithm in algorithms:
        if 'mordor' in environment:
            # Read the table
            df = pd.read_csv(f'../mordor_results/summary/{environment}-{algorithm}.txt', delimiter='\t', 
                            names=['timestep', '25th', 'median', '75th'])
        else:
            df = pd.read_csv(f'../test_results/summary/{environment}-{algorithm}.txt', delimiter='\t', 
                            names=['timestep', '25th', 'median', '75th'])
            
        main_df.loc[0, algorithm] = df.iloc[-1].to_dict()['median']
        df_25th.loc[0, algorithm] = float(df.iloc[-1]['25th'])
        df_75th.loc[0, algorithm] = float(df.iloc[-1]['75th'])
    for noise in range(10, 99, 10):
        for algorithm in algorithms:
            if 'mordor' in environment:
                # Read the table
                df = pd.read_csv(f'../mordor_noise_results/{noise}/summary/{environment}-{algorithm}.txt', delimiter='\t', 
                                names=['timestep', '25th', 'median', '75th'])
            else:
                # Read the table
                df = pd.read_csv(f'../noise_results/{noise}/summary/{environment}-{algorithm}.txt', delimiter='\t', 
                                names=['timestep', '25th', 'median', '75th'])
            
            main_df.loc[noise // 10, algorithm] = df.iloc[-1].to_dict()['median']
            df_25th.loc[noise // 10, algorithm] = float(df.iloc[-1]['25th'])
            df_75th.loc[noise // 10, algorithm] = float(df.iloc[-1]['75th'])

           
    # Plotting the data
    plt.figure(figsize=(10, 6))
    for algorithm in algorithms:
        plt.plot(main_df.index * 10, main_df[algorithm], marker='o', label=algorithm.upper())
        plt.fill_between(df_25th.index * 10, df_25th[algorithm], df_75th[algorithm], alpha=0.2)
    
    plt.xlabel('Noise level')
    plt.ylabel('Avg. reward in the 100000th step')
    plt.title(f'Performance of algorithms in {environment}')
    plt.xlim(0, 90)
    plt.xticks([i * 10 for i in range(10)])
    plt.legend()
    plt.grid(True)
    plt.savefig(f'../noise_results/noise_plots/{environment}_performance_plot.png')
    plt.show()

        