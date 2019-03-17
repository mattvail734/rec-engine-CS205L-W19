import pandas as pd
from surprise import Reader, Dataset
from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor
from surprise import KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import BaselineOnly, CoClustering
from surprise.model_selection import cross_validate
import load_data
import custom_matrix_factorization
import save_to_output

# for more information on loading data to surprise:
# https://surprise.readthedocs.io/en/stable/getting_started.html#load-custom


def benchmark():
    ratings = load_data.load_small_ratings()
    # ratings = clean_data.filter_by_rating_count(ratings)
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
    save_to_output.store_dataframe(output, 'Algorithm_Benchmark.csv')


# THIS DOESN'T WORK YET. Complete matrix function doesn't work. Need to fix that first.
def custom():
    ratings = load_data.load_small_ratings()
    # ratings = clean_data.filter_by_rating_count(ratings)
    ratings = load_data.complete_ratings_matrix(ratings)

    K = 20
    alpha = 0.001
    beta = 0.01
    iterations = 100

    custom_mf = custom_matrix_factorization.custom_matrix_factorization(ratings, K=K, alpha=alpha, beta=beta, iterations=iterations)
    custom_mf.train()
    text_file = open('custom_performance.txt', 'w')
    text_file.write('K: {}'.format(K)+'\n')
    text_file.write('alpha: {}'.format(alpha)+'\n')
    text_file.write('beta: {}'.format(beta)+'\n')
    text_file.write('iterations: {}'.format(iterations)+'\n')
    text_file.write('RMSE: {}'.format(custom_mf.rmse())+'\n')
    text_file.close()
    save_to_output.move_to_output('custom_performance.txt')


if __name__ == '__main__':
    custom()
