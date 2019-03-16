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
