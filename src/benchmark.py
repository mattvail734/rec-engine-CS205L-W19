import os
import pandas as pd
from surprise import Reader, Dataset
from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor
from surprise import KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import BaselineOnly, CoClustering
from surprise.model_selection import cross_validate
import load_data
import clean
import custom_matrix_factorization

# for more information on loading data to surprise:
# https://surprise.readthedocs.io/en/stable/getting_started.html#load-custom


def benchmark():
    ratings = load_data.load_ratings()
    ratings = clean.filter_by_rating_count(ratings)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    benchmark = []
    algorithms = [
        SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(),
        KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]

    for algorithm in algorithms:
        results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
        temp = pd.DataFrame.from_dict(results).mean(axis=0)
        temp = temp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
        benchmark.append(temp)

    output = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')
    store_dataframe(output, 'Algorithm_Benchmark.csv')


def store_dataframe(df, file_name):
    dirname = os.path.dirname(__file__)
    output_path = os.path.join(dirname, file_name)
    df.to_csv(output_path, index=True, encoding='utf-8')


def custom():
    ratings = load_data.load_ratings()
    ratings = clean.filter_by_rating_count(ratings)
    ratings = clean.complete_ratings_matrix(ratings)

    K = 20
    alpha = 0.001
    beta = 0.01
    iterations = 100

    custom_mf = custom_matrix_factorization(ratings, K=K, alpha=alpha, beta=beta, iterations=iterations)
    training_process = custom_mf.train()
    text_file = open('custom_perormance.txt', 'w')
    text_file.write('K: {}'.format(K))
    text_file.write('alpha: {}'.format(alpha))
    text_file.write('beta: {}'.format(beta))
    text_file.write('iterations: {}'.format(iterations))
    text_file.write('RMSE: {}'.format(custom_mf.rmse()))
    text_file.write(training_process)
    text_file.write('P x Q:')
    text_file.write(custom_mf.full_matrix())


if __name__ == '__main__':
    custom()
