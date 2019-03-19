import numpy as np
from numpy.linalg import norm
from surprise import SVD, SVDpp, NMF
from surprise.utils import get_rng
from scipy.optimize import minimize
# https://surprise.readthedocs.io/en/stable/building_custom_algo.html
# https://github.com/NicolasHug/Surprise/blob/711fb80748140c44e0ed870e573c735307e6c3cc/surprise/prediction_algorithms/matrix_factorization.pyx
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_cg.html


# class extends the SVD algorithm to use Conjugate Gradient Descent
class SVD_CG(SVD):

    def fit(self, trainset):
        SVD.fit(self, trainset)
        self.CG(trainset)
        return self

    def CG(self, trainset):
        lr_pu = self.lr_pu
        lr_qi = self.lr_qi
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi

        # define global_mean, regularization parameters, and initialize baselines to 0
        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)
        global_mean = 0
        lr_bu = 0
        lr_bi = 0
        reg_bu = 0
        reg_bi = 0
        if self.biased:
            global_mean = self.trainset.global_mean
            # lr_bu = self.lr_bu
            # lr_bi = self.lr_bi
            reg_bu = self.reg_bu
            reg_bi = self.reg_bi

        # initialize factors randomly
        rng = get_rng(self.random_state)
        pu = rng.normal(self.init_mean, self.init_std_dev, (trainset.n_users, self.n_factors))
        qi = rng.normal(self.init_mean, self.init_std_dev, (trainset.n_items, self.n_factors))

        # this currently just does SGD
        # I think it's the only thing that needs to be modified to do CG
        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))
            # u = user, i = item = movie, r = rating
            # complete ratings matrix is never computed
            sk1 = 1
            res_k1 = 1
            for u, i, r in trainset.all_ratings():
                res_k = res_k1
                res_k1 = global_mean + bu[u] + bi[i] + np.dot(pu[u, :], qi[i, :])
                sk = sk1

                # update factors
                pu[u, :] += lr_pu * (res_k * qi[i, :] - reg_pu * pu[u, :])
                qi[i, :] += lr_qi * (res_k * pu[u, :] - reg_qi * qi[i, :])

                # update biases
                if self.biased:
                    ak = (res_k * res_k) / (sk * max(bu[u], 0.1) * sk)
                    sk1 = res_k1 + (res_k1 * res_k1) / (res_k * res_k) * sk
                    bu[u] += ak * sk1
                    bi[i] += ak * sk1

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi


# class extends the SVDpp algorithm to use Conjugate Gradient Descent
class SVDpp_CG(SVDpp):

    def fit(self, trainset):
        SVDpp.fit(self, trainset)
        self.CG(trainset)
        return self

# conjugate gradient descent algorithm (FIND SOURCE)
    def CG(self, trainset):
        # implement conjugate gradient descent here
        return


# class extends the SVDpp algorithm to use Conjugate Gradient Descent
class NMF_CG(NMF):

    def fit(self, trainset):
        NMF.fit(self, trainset)
        self.CG(trainset)
        return self

# conjugate gradient descent algorithm (FIND SOURCE)
    def CG(self, trainset):
        # implement conjugate gradient descent here
        return
