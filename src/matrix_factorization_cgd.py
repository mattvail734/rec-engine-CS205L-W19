import numpy as np
from surprise import SVD, SVDpp, NMF
from surprise.utils import get_rng
# https://surprise.readthedocs.io/en/stable/building_custom_algo.html
# https://github.com/NicolasHug/Surprise/blob/711fb80748140c44e0ed870e573c735307e6c3cc/surprise/prediction_algorithms/matrix_factorization.pyx
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_cg.html


# class extends the SVD algorithm to use Conjugate Gradient Descent
class SVD_cgd(SVD):

    def fit(self, trainset):
        SVD.fit(self, trainset)
        self.cgd(trainset)
        return self

    def cgd(self, trainset):
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
            lr_bu = self.lr_bu
            lr_bi = self.lr_bi
            reg_bu = self.reg_bu
            reg_bi = self.reg_bi

        # initialize factors randomly
        rng = get_rng(self.random_state)
        pu = rng.normal(self.init_mean, self.init_std_dev, (trainset.n_users, self.n_factors))
        qi = rng.normal(self.init_mean, self.init_std_dev, (trainset.n_items, self.n_factors))

        # this currently just does SGD
        # I think it's the only thing that needs to be modified to do CGD
        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))
            # u = user, i = item = movie, r = rating
            # complete ratings matrix is never computed
            for u, i, r in trainset.all_ratings():
                # compute current error
                residual = r - (global_mean + bu[u] + bi[i] + np.dot(qi[i, :], pu[u, :]))

                # update factors
                pu[u, :] += lr_pu * (residual * qi[i, :] - reg_pu * pu[u, :])
                qi[i, :] += lr_qi * (residual * pu[u, :] - reg_qi * qi[i, :])

                # update biases
                if self.biased:
                    bu[u] += lr_bu * (residual - reg_bu * bu[u])
                    bi[i] += lr_bi * (residual - reg_bi * bi[i])

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi


# class extends the SVDpp algorithm to use Conjugate Gradient Descent
class SVDpp_cgd(SVDpp):

    def fit(self, trainset):
        SVDpp.fit(self, trainset)
        self.cgd(trainset)
        return self

# conjugate gradient descent algorithm (FIND SOURCE)
    def cgd(self, trainset):
        # implement conjugate gradient descent here
        return


# class extends the SVDpp algorithm to use Conjugate Gradient Descent
class NMF_cgd(NMF):

    def fit(self, trainset):
        NMF.fit(self, trainset)
        self.cgd(trainset)
        return self

# conjugate gradient descent algorithm (FIND SOURCE)
    def cgd(self, trainset):
        # implement conjugate gradient descent here
        return
