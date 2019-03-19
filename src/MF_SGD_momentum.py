import numpy as np
from numpy.linalg import norm
from surprise import SVD, SVDpp, NMF
from surprise.utils import get_rng
# https://surprise.readthedocs.io/en/stable/building_custom_algo.html
# https://github.com/NicolasHug/Surprise/blob/711fb80748140c44e0ed870e573c735307e6c3cc/surprise/prediction_algorithms/matrix_factorization.pyx
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_SGD_momentum.html


# class extends the SVD algorithm to add a momentum component to the SGD
class SVD_SGD_momentum(SVD):

    def fit(self, trainset):
        SVD.fit(self, trainset)
        self.SGD_momentum(trainset)
        return self

    def SGD_momentum(self, trainset):
        momenum_all = 0.1
        lr_pu = self.lr_pu
        lr_qi = self.lr_qi
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi
        mom_pu = momenum_all
        mom_qi = momenum_all

        # define global_mean, regularization parameters, and initialize baselines to 0
        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)
        global_mean = 0
        lr_bu = 0
        lr_bi = 0
        reg_bu = 0
        reg_bi = 0
        mom_bu = momenum_all
        mom_bi = momenum_all
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

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))
            # u = user, i = item = movie, r = rating
            v_bu = 0
            v_bi = 0
            v_pu = 0
            v_qi = 0
            for u, i, r in trainset.all_ratings():
                # compute current residual
                residual = r - global_mean - bu[u] - bi[i] - np.dot(pu[u, :], qi[i, :])

                # update biases
                if self.biased:
                    v_bu_prior = v_bu
                    v_bu = mom_bu * v_bu_prior + (1 - mom_bu) * residual
                    bu[u] += lr_bu * (v_bu - reg_bu * bu[u])
                    v_bi_prior = v_bi
                    v_bi = mom_bi * v_bi_prior + (1 - mom_bi) * residual
                    bi[i] += lr_bi * (v_bi - reg_bi * bi[i])

                # update factors
                v_pu_prior = v_pu
                v_pu = mom_pu * v_pu_prior + (1 - mom_pu) * residual
                pu[u, :] += lr_pu * (v_pu * qi[i, :] - reg_pu * pu[u, :])
                v_qi_prior = v_qi
                v_qi = mom_qi * v_qi_prior + (1 - mom_qi) * residual
                qi[i, :] += lr_qi * (v_qi * pu[u, :] - reg_qi * qi[i, :])

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi


# class extends the SVDpp algorithm to add a momentum component to the SGD
class SVDpp_SGD_momentum(SVDpp):

    def fit(self, trainset):
        SVDpp.fit(self, trainset)
        self.SGD_momentum(trainset)
        return self

    def SGD_momentum(self, trainset):
        momenum_all = 0.1
        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        lr_pu = self.lr_pu
        lr_qi = self.lr_qi
        lr_yj = self.lr_yj
        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi
        reg_yj = self.reg_yj
        mom_pu = momenum_all
        mom_qi = momenum_all
        mom_yj = momenum_all
        mom_bu = momenum_all
        mom_bi = momenum_all

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)
        global_mean = self.trainset.global_mean

        # initialize factors randomly
        rng = get_rng(self.random_state)
        pu = rng.normal(self.init_mean, self.init_std_dev, (trainset.n_users, self.n_factors))
        qi = rng.normal(self.init_mean, self.init_std_dev, (trainset.n_items, self.n_factors))
        yj = rng.normal(self.init_mean, self.init_std_dev, (trainset.n_items, self.n_factors))

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print(" processing epoch {}".format(current_epoch))
            v_bu = 0
            v_bi = 0
            v_pu = 0
            v_qi = 0
            v_yj = 0
            # u = user, i = item = movie, r = rating
            for u, i, r in trainset.all_ratings():
                # items rated by u. This is COSTLY
                Iu = [j for (j, _) in trainset.ur[u]]
                sqrt_Iu = np.sqrt(len(Iu))

                # compute user implicit feedback
                u_impl_fdb = np.zeros(self.n_factors, np.double)
                for j in Iu:
                    for f in range(self.n_factors):
                        u_impl_fdb[f] += yj[j, f] / sqrt_Iu

                # compute current residual
                residual = r - global_mean - bu[u] - bi[i] - np.dot(pu[u, :], qi[i, :]-u_impl_fdb[f])

                # update biases
                v_bu_prior = v_bu
                v_bu = mom_bu * v_bu_prior + (1 - mom_bu) * residual
                bu[u] += lr_bu * (v_bu - reg_bu * bu[u])
                v_bi_prior = v_bi
                v_bi = mom_bi * v_bi_prior + (1 - mom_bi) * residual
                bi[i] += lr_bi * (v_bi - reg_bi * bi[i])

                # update factors
                v_pu_prior = v_pu
                v_pu = mom_pu * v_pu_prior + (1 - mom_pu) * residual
                pu[u, :] += lr_pu * (v_pu * qi[i, :] - reg_pu * pu[u, :])
                v_qi_prior = v_qi
                v_qi = mom_qi * v_qi_prior + (1 - mom_qi) * residual
                qi[i, :] += lr_qi * (v_qi * pu[u, :] - reg_qi * qi[i, :])
                v_yj_prior = v_yj
                v_yj = mom_yj * v_yj_prior + (1 - mom_yj) * residual
                yj[j, :] += lr_yj * (v_yj * qi[i, :] / sqrt_Iu - reg_yj * yj[j, :])

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
        self.yj = yj
