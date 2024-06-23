import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


environments = ['office', 'office-single']
algorithms = ['crm', 'crm-rs', 'hrm', 'hrm-rs', 'ql', 'ql-rs']

for noise in range(10, 99, 10):
    for environment in environments:
        for algorithm in algorithms:
            # Read the table
            df = pd.read_csv(f'../noise_results/{noise}/summary/{environment}-{algorithm}.txt', delimiter='\t', 
                             names=['timestep', '25th', 'median', '75th'])

            # Plot median reward as a line
            plt.plot(df['timestep'], df['median'], label=f'{algorithm.upper()}')

            # Plot shaded area between 25th and 75th percentile rewards
            plt.fill_between(df['timestep'], df['25th'], df['75th'], alpha=0.3)


        # Add labels, ticks and legend
        plt.title(f'{environment} with {noise}% noise')
        plt.xlabel('Training steps (in thousands)')
        plt.ylabel('Avg. reward per step')
        plt.ylim(0, 1.3)
        plt.yticks([i / 10 for i in range(13)])
        plt.xlim(0, 100)
        plt.xticks([i for i in range(0, 101, 10)])
        plt.legend()

        # Save plot
        plt.savefig(f'../noise_results/{noise}/plots/{environment}.png')
        
        # Show plot
        plt.show()
        plt.clf()