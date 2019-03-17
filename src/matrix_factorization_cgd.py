from surprise import AlgoBase
# https://surprise.readthedocs.io/en/stable/building_custom_algo.html
# https://github.com/NicolasHug/Surprise/blob/711fb80748140c44e0ed870e573c735307e6c3cc/surprise/prediction_algorithms/matrix_factorization.pyx


# class extends the SVD algorithm to use Conjugate Gradient Descent
class matrix_factorization_cgd_svd(AlgoBase.SVD):

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.cgd(trainset)
        return self

# conjugate gradient descent algorithm (FIND SOURCE)
    def cgd(self, trainset):
        # implement conjugate gradient descent here
        return


# class extends the SVDpp algorithm to use Conjugate Gradient Descent
class matrix_factorization_cgd_svdpp(AlgoBase.SVDpp):

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.cgd(trainset)
        return self

# conjugate gradient descent algorithm (FIND SOURCE)
    def cgd(self, trainset):
        # implement conjugate gradient descent here
        return
