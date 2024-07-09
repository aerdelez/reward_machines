import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

environments = ['office', 'office-single']
algorithms = ['CRM', 'CRM-RS', 'HRM', 'HRM-RS', 'QL', 'QL-RS']

for environment in environments:
    for algorithm in algorithms:
        # Read the table
        df = pd.read_csv(f'../difference_results/{environment}/{algorithm}.txt', delimiter='\t') 

        # Calculate the difference and average
        difference = df['median']
        average = df['means']

        # Bland-Altman plot
        plt.figure(figsize=(8, 6))
        plt.scatter(average, difference, color='blue', alpha=0.5)
        plt.axhline(np.mean(difference), color='red', linestyle='--', label='Mean difference')
        plt.axhline(np.mean(difference) + 1.96 * np.std(difference), color='gray', linestyle='--', label='95% limits of agreement')
        plt.axhline(np.mean(difference) -1.96 * np.std(difference), color='gray', linestyle='--')
        plt.xlabel('Average of replicated and original medians')
        plt.ylabel('Difference (replicated - original)')
        plt.title(f'Bland-Altman plot for {algorithm} in {environment}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'../difference_results/{environment}/{algorithm}.png')
        plt.show()

