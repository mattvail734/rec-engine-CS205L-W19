import os
import numpy as np
import pandas as pd
import math
import re
from scipy.sparse import csr_matrix
from surprise import Reader, Dataset, SVD, evaluate
import matplotlib.pyplot as plt
import seaborn as sns

# download the ml-20m dataset from
# http://files.grouplens.org/datasets/movielens/ml-20m.zip
# unzip and place the entire 'ml-20m' folder in repo (not tracked by git) before running

sns.set_style('darkgrid')

genome_scores = pd.read_csv('../ml-20m/genome-scores.csv')
genome_tags = pd.read_csv('../ml-20m/genome-tags.csv')
links = pd.read_csv('../ml-20m/links.csv')
movies = pd.read_csv('../ml-20m/movies.csv')
ratings = pd.read_csv('../ml-20m/ratings.csv')
tags = pd.read_csv('../ml-20m/tags.csv')

print(genome_scores.head())
print(genome_tags.head())
print(links.head())
print(movies.head())
print(ratings.head())
print(tags.head())

print(genome_scores.isnull().any())
print(genome_tags.isnull().any())
print(links.isnull().any())
print(movies.isnull().any())
print(ratings.isnull().any())
print(tags.isnull().any())

print(genome_scores.columns)
print(genome_tags.columns)
print(links.columns)
print(movies.columns)
print(ratings.columns)
print(tags.columns)
