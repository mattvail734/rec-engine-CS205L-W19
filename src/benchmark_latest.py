import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset
from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor
from surprise import KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import BaselineOnly, CoClustering
from surprise.model_selection import cross_validate, GridSearchCV
import load_data
from MF_SGD_momentum import SVD_SGD_momentum, SVDpp_SGD_momentum

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
    i = 0
    while os.path.exists(output_path):
        output_path = os.path.join(output_dir, str(i) + file_name)
        i += 1
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


def surprise_latest_dataset():
    ratings = load_data.load_latest_ratings()
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    return data


def surprise_latest_dataset_filtered_by_user():
    ratings = load_data.load_latest_ratings()
    ratings = load_data.filter_users_by_rating_count(ratings)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    return data


def surprise_latest_dataset_filtered_by_movie():
    ratings = load_data.load_latest_ratings()
    ratings = load_data.filter_movies_by_rating_count(ratings)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    return data


def surprise_latest_dataset_filtered_by_both():
    ratings = load_data.load_latest_ratings()
    ratings = load_data.filter_by_rating_count(ratings)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    return data


def benchmark(data):
    performance = []
    algorithms = [
        SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(),
        KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(),
        CoClustering(), SVD_SGD_momentum(), SVDpp_SGD_momentum()]
    for algorithm in algorithms:
        results = cross_validate(algorithm, data, measures=['RMSE', 'MAE', 'FCP'], cv=3, verbose=False)
        output = pd.DataFrame.from_dict(results).mean(axis=0)
        output = output.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
        performance.append(output)
    output_df = pd.DataFrame(performance).set_index('Algorithm').sort_values('test_rmse')
    store_dataframe(output_df, 'Algorithm_Benchmark.csv')


def parameter_search(data):
    param_grid = {
        'n_epochs': [1, 2, 3, 4, 5, 10, 20, 40, 60, 80, 100],
        'lr_all': [.002, .005, .01, .02],
        'reg_all': [0.2, 0.6, 1.0],
        'init_mean': [0.0, 0.5, 1.0],
        'init_std_dev': [0.0, 0.1, 0.5, 1.0]}
    algorithms = [SVD, SVD_SGD_momentum, SVDpp_SGD_momentum]
    for algorithm in algorithms:
        gs = GridSearchCV(SVD, param_grid)
        gs.fit(data)
        output_df = pd.DataFrame.from_dict(gs.cv_results)
        file_name = str(algorithm).split(' ')[1].split('.')[-1].split('\'')[0] + '_parameter_search_.csv'
        store_dataframe(output_df, file_name)


# This should output 4 "Algorithm Benchmark" csv files:
# 'Algorithm_Benchmark.csv' - unfiltered latest data set
# '0Algorithm_Benchmark.csv' - user filtered latest data set
# '1Algorithm_Benchmark.csv' - movie filtered latest data set
# '2Algorithm_Benchmark.csv' - both filtered latest data set
# as well as 12 (4 data sets x 3 algorithms) "Parameter Search" csv files:
# 'SVD_parameter_search.csv' - unfiltered latest data set
# '0SVD_parameter_search.csv' - user filtered latest data set
# '1SVD_parameter_search.csv' - movie filtered latest data set
# '2SVD_parameter_search.csv' - both filtered latest data set
# 'SVD_SGD_momentum.csv' - unfiltered latest data set
# '0SVD_SGD_momentum.csv' - user filtered latest data set
# '1SVD_SGD_momentum.csv' - movie filtered latest data set
# '2SVD_SGD_momentum.csv' - both filtered latest data set
# 'SVDpp_SGD_momentum.csv' - unfiltered latest data set
# '0SVDpp_SGD_momentum.csv' - user filtered latest data set
# '1SVDpp_SGD_momentum.csv' - movie filtered latest data set
# '2SVDpp_SGD_momentum.csv' - both filtered latest data set

if __name__ == '__main__':
    data = surprise_latest_dataset_filtered_by_both()
    benchmark(data)
    parameter_search(data)
    user_filtered_data = surprise_latest_dataset_filtered_by_user()
    benchmark(user_filtered_data)
    parameter_search(user_filtered_data)
    movie_filtered_data = surprise_latest_dataset_filtered_by_movie()
    benchmark(movie_filtered_data)
    parameter_search(movie_filtered_data)
    both_filtered_data = surprise_latest_dataset_filtered_by_both()
    benchmark(both_filtered_data)
    parameter_search(both_filtered_data)
