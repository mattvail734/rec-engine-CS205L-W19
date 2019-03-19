import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset
from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor
from surprise import KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import BaselineOnly, CoClustering
from surprise.model_selection import cross_validate
import load_data
from matrix_factorization_cgd import SVD_cgd

# for more information on loading data to surprise:
# https://surprise.readthedocs.io/en/stable/getting_started.html#load-custom


def plot_benchmark():
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
        save_fig(plt, 'Surprise_Algorithms_Benchmark_RMSE')


def store_dataframe(df, file_name):
    dirname = os.path.dirname(__file__)
    output_dir = os.path.relpath('../output', dirname)
    output_path = os.path.join(output_dir, file_name)
    df.to_csv(output_path, index=True, encoding='utf-8')


def save_fig(plt, file_name):
    dirname = os.path.dirname(__file__)
    output_dir = os.path.relpath('../output', dirname)
    output_path = os.path.join(output_dir, file_name)
    plt.savefig(output_path)


def move_to_output(file_name):
    current_dir = os.path.dirname(__file__)
    current_path = os.path.join(current_dir, file_name)
    output_dir = os.path.relpath('../output', current_dir)
    output_path = os.path.join(output_dir, file_name)
    os.rename(current_path, output_path)


def benchmark():
    ratings = load_data.load_small_ratings()
    # ratings = clean_data.filter_by_rating_count(ratings)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    benchmark = []
    algorithms = [SVD_cgd(n_factors=10, n_epochs=5)]
    # algorithms = [
        # SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(),
        # KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering(), SVD_cgd(n_factors=10, n_epochs=5, biased=False)]

    for algorithm in algorithms:
        results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
        temp = pd.DataFrame.from_dict(results).mean(axis=0)
        temp = temp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
        benchmark.append(temp)

    output = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')
    store_dataframe(output, 'Algorithm_Benchmark.csv')
    plot_benchmark()


if __name__ == '__main__':
    benchmark()
