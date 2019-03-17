import pandas as pd


# filter dataset by number of reviews
# returns the filtered dataframe
def filter_by_rating_count(ratings, percentile=0.999):
    stats = ['count', 'mean']

    ratings_summary_users = ratings.groupby('userId')['rating'].agg(stats)
    ratings_summary_users.index = ratings_summary_users.index.map(int)
    min_ratings_user = round(ratings_summary_users['count'].quantile(percentile), 0)
    drop_user_list = ratings_summary_users[ratings_summary_users['count'] < min_ratings_user].index

    ratings_summary_movies = ratings.groupby('movieId')['rating'].agg(stats)
    ratings_summary_movies.index = ratings_summary_movies.index.map(int)
    min_ratings_movie = round(ratings_summary_movies['count'].quantile(percentile), 0)
    drop_movie_list = ratings_summary_movies[ratings_summary_movies['count'] < min_ratings_movie].index

    ratings_short = ratings[~ratings['movieId'].isin(drop_movie_list)]
    ratings_short = ratings_short[~ratings_short['userId'].isin(drop_user_list)]
    return ratings_short


# remove all nan values
# creates the complete ratings matrix from a dataframe of ratings with
# user x movie
def complete_ratings_matrix(ratings):
    complete_ratings = pd.pivot_table(ratings, values='rating', index='userId', columns='movieId')
    complete_ratings = complete_ratings.fillna(0)
    complete_ratings_matrix = complete_ratings.to_numpy()
    return complete_ratings_matrix
