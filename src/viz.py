import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_style('darkgrid')

algorithm_benchmark = pd.read_csv('Algorithm_Benchmark.csv', header=0, index_col='Algorithm')
bar = sns.barplot(x=algorithm_benchmark.index, y=algorithm_benchmark['test_rmse'], palette='rocket')
bar.set_ylabel('Root Mean Squared Error (RMSE)')
bar.set_title('RMSE on Movie Rating Prediction Across Various Models')
for tick in bar.get_xticklabels():
    tick.set_rotation(20)
plt.savefig('Surprise_Algorithms_Benchmark_RMSE.png')
