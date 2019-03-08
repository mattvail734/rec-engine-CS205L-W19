import numpy as np
import pandas as pd
import math
import re
from scipy.sparse import csr_matrix
from surprise import Reader, Dataset, SVD, evaluate
import matplotlib.pyplot as plt
import seaborn as sns


# download the netflix prize datatset from
# https://www.kaggle.com/netflix-inc/netflix-prize-data/downloads/netflix-prize-data.zip/1
# unzip and place the entire 'netflix-prize-data' folder in repo (not tracked by git) before running

sns.set_style('darkgrid')

df1 = pd.read_csv('../netflix-prize-data/combined_data_1.txt')
print(df1.head())
