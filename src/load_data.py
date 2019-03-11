import pandas as pd

# download the ml-20m dataset from
# http://files.grouplens.org/datasets/movielens/ml-20m.zip
# unzip and place the entire 'ml-20m' folder in repo (not tracked by git) before running

# for more information on the data:
# http://files.grouplens.org/datasets/movielens/ml-20m-README.html


# movieId,tagId,relevance
def load_genome_scores():
    genome_scores = pd.read_csv('../ml-20m/genome-scores.csv')
    return genome_scores


# tagId,tag
def load_genome_tags():
    genome_tags = pd.read_csv('../ml-20m/genome-tags.csv')
    return genome_tags


# movieId,imdbId,tmdbId
def load_links():
    links = pd.read_csv('../ml-20m/links.csv')
    return links


# movieId,title,genres
def load_movies():
    movies = pd.read_csv('../ml-20m/movies.csv')
    return movies


# userId,movieId,rating,timestamp
def load_ratings():
    ratings = pd.read_csv('../ml-20m/ratings.csv')
    return ratings


# userId,movieId,tag,timestamp
def load_tags():
    tags = pd.read_csv('../ml-20m/tags.csv')
    return tags
