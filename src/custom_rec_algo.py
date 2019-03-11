import numpy as np
from surprise import AlgoBase
from surprise import Dataset
from surprise.model_selection import cross_validate, GridSearchCV
# https://surprise.readthedocs.io/en/stable/building_custom_algo.html


# this is a very rough draft based on the surprise tutorial for customizing your
# own algorithm; I also implemented a simple gradient descent based factorization
# but now I need to do more research on how to avoid computing the complete ratings matrix
class custom_rec_algo(AlgoBase):
    def __init__(self, sim_options={}, bsl_options={}):
        AlgoBase.__init__(self, sim_options=sim_options, bsl_options=bsl_options)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.the_mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])
        self.bu, self.bi = self.compute_baselines()
        self.sim = self.compute_similarities()
        return self

    def estimate(self, u, i):
        sum_means = self.trainset.global_mean
        div = 1

        if self.trainset.knows_user(u):
            sum_means += np.mean([r for (_, r) in self.trainset.ur[u]])
            div += 1
        if self.trainset.knows_item(i):
            sum_means += np.mean([r for (_, r) in self.trainset.ir[i]])
            div += 1

        details = {'info1': 'that was', 'info2': 'easy stuff'}
        return sum_means/div, details


data = Dataset.load_builtin('ml-100k')
algo = custom_rec_algo

param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005], 'reg_all': [0.4, 0.6]}
gs = GridSearchCV(custom_rec_algo, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)

cross_validate(algo, data, verbose=True)
