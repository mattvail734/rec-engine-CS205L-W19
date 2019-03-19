import pandas as pd

# download the ml-20m dataset from
# http://files.grouplens.org/datasets/movielens/ml-20m.zip
# unzip and place the entire 'ml-20m' folder in repo (not tracked by git) before running

# for more information on the data:
# http://files.grouplens.org/datasets/movielens/ml-20m-README.html


# movieId,tagId,relevance
def load_20m_genome_scores():
    genome_scores = pd.read_csv('../ml-20m/20m/genome-scores.csv')
    return genome_scores


# tagId,tag
def load_20m_genome_tags():
    genome_tags = pd.read_csv('../ml-20m/20m/genome-tags.csv')
    return genome_tags


# movieId,imdbId,tmdbId
def load_20m_links():
    links = pd.read_csv('../ml-20m/20m/links.csv')
    return links


# movieId,title,genres
def load_20m_movies():
    movies = pd.read_csv('../ml-20m/20m/movies.csv')
    return movies


# userId,movieId,rating,timestamp
def load_20m_ratings():
    ratings = pd.read_csv('../ml-20m/20m/ratings.csv')
    return ratings


# userId,movieId,tag,timestamp
def load_20m_tags():
    tags = pd.read_csv('../ml-20m/ml-latest/tags.csv')
    return tags


# movieId,tagId,relevance
def load_latest_genome_scores():
    genome_scores = pd.read_csv('../ml-20m/ml-latest/genome-scores.csv')
    return genome_scores


# tagId,tag
def load_latest_genome_tags():
    genome_tags = pd.read_csv('../ml-20m/ml-latest/genome-tags.csv')
    return genome_tags


# movieId,imdbId,tmdbId
def load_latest_links():
    links = pd.read_csv('../ml-20m/ml-latest/links.csv')
    return links


# movieId,title,genres
def load_latest_movies():
    movies = pd.read_csv('../ml-20m/ml-latest/movies.csv')
    return movies


# userId,movieId,rating,timestamp
def load_latest_ratings():
    ratings = pd.read_csv('../ml-20m/ml-latest/ratings.csv')
    return ratings


# userId,movieId,tag,timestamp
def load_latest_tags():
    tags = pd.read_csv('../ml-20m/ml-latest/tags.csv')
    return tags


# movieId,tagId,relevance
def load_small_genome_scores():
    genome_scores = pd.read_csv('../ml-20m/ml-latest-small/genome-scores.csv')
    return genome_scores


# tagId,tag
def load_small_genome_tags():
    genome_tags = pd.read_csv('../ml-20m/ml-latest-small/genome-tags.csv')
    return genome_tags


# movieId,imdbId,tmdbId
def load_small_links():
    links = pd.read_csv('../ml-20m/ml-latest-small/links.csv')
    return links


# movieId,title,genres
def load_small_movies():
    movies = pd.read_csv('../ml-20m/ml-latest-small/movies.csv')
    return movies


# userId,movieId,rating,timestamp
def load_small_ratings():
    ratings = pd.read_csv('../ml-20m/ml-latest-small/ratings.csv')
    return ratings


# userId,movieId,tag,timestamp
def load_small_tags():
    tags = pd.read_csv('../ml-20m/ml-latest-small/tags.csv')
    return tags


# filter dataset by number of reviews
# returns the filtered dataframe
# order matters, if filtering both users and movies use filter_by_rating_count
def filter_by_rating_count(ratings, percentile=0.99):
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


# filter users in the dataset by number of reviews
# returns the filtered dataframe
def filter_users_by_rating_count(ratings, percentile=0.999):
    stats = ['count', 'mean']
    ratings_summary_users = ratings.groupby('userId')['rating'].agg(stats)
    ratings_summary_users.index = ratings_summary_users.index.map(int)
    min_ratings_user = round(ratings_summary_users['count'].quantile(percentile), 0)
    drop_user_list = ratings_summary_users[ratings_summary_users['count'] < min_ratings_user].index
    ratings_short = ratings[~ratings['userId'].isin(drop_user_list)]
    return ratings_short

# filter movies in the dataset by number of reviews
# returns the filtered dataframe
def filter_movies_by_rating_count(ratings, percentile=0.999):
    stats = ['count', 'mean']
    ratings_summary_movies = ratings.groupby('movieId')['rating'].agg(stats)
    ratings_summary_movies.index = ratings_summary_movies.index.map(int)
    min_ratings_movie = round(ratings_summary_movies['count'].quantile(percentile), 0)
    drop_movie_list = ratings_summary_movies[ratings_summary_movies['count'] < min_ratings_movie].index
    ratings_short = ratings[~ratings['movieId'].isin(drop_movie_list)]
    return ratings_short


# remove all nan values
# creates the complete ratings matrix from a dataframe of ratings with
# user x movie
def complete_ratings_matrix(ratings):
    complete_ratings = pd.pivot_table(ratings, values='rating', index='userId', columns='movieId')
    complete_ratings = complete_ratings.fillna(0)
    complete_ratings_matrix = complete_ratings.to_numpy()
    return complete_ratings_matrix
