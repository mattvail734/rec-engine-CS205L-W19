import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import save_to_output

# sns.set_style('darkgrid')
file_name = 'Algorithm_Benchmark.csv'
dirname = os.path.dirname(__file__)
input_dir = os.path.relpath('../output', dirname)
input_path = os.path.join(input_dir, file_name)
algorithm_benchmark = pd.read_csv(input_path, header=0, index_col='Algorithm')
bar = sns.barplot(x=algorithm_benchmark.index, y=algorithm_benchmark['test_rmse'], palette='rocket')
bar.set_ylabel('Root Mean Squared Error (RMSE)')
bar.set_title('RMSE on Movie Rating Prediction Across Various Models')
for tick in bar.get_xticklabels():
    tick.set_rotation(20)
save_to_output.save_fig(plt, 'Surprise_Algorithms_Benchmark_RMSE')
