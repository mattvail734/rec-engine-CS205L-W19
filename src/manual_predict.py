import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import load_data
import clean_data


def calc_user_similarity(complete_ratings_matrix):
    return pairwise_distances(complete_ratings_matrix, metric='cosine')


def calc_movie_similarity(complete_ratings_matrix):
    return pairwise_distances(complete_ratings_matrix.T, metric='cosine')


def predict(complete_ratings_matrix, type='user'):
    user_similarity = calc_user_similarity(complete_ratings_matrix)
    movie_similarity = calc_movie_similarity(complete_ratings_matrix)
    if type == 'user':
        mean_user_rating = complete_ratings_matrix.mean(axis=1)
        ratings_difference = (complete_ratings_matrix - mean_user_rating[:, np.newaxis])
        prediction = mean_user_rating[:, np.newaxis] + user_similarity.dot(
            ratings_difference)/np.array([np.abs(user_similarity).sum(axis=1)]).T
    elif type == 'movie':
        prediction = complete_ratings_matrix.dot(movie_similarity) / np.array(
            [np.abs(movie_similarity).sum(axis=1)])
    return prediction


if __name__ == '__main__':
    ratings = load_data.load_small_ratings()
    complete_ratings_matrix = clean_data.complete_ratings_matrix(ratings)
    predictions = predict(complete_ratings_matrix)
    print(predictions)
