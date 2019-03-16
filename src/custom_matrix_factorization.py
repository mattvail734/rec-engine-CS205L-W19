import numpy as np


class custom_matrix_factorization():

    # R = ratings matrix (users x movies)
    # K = number of latent features
    # alpha = learning rate
    # beta = regularization parameter for bias
    # iterations = learning steps in gradient descent

    # initializing the user-movie rating matrix
    def __init__(self, R, K, alpha, beta, iterations):
        self.R = R
        self.num_users, self.num_movies = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # initializing the user-feature and movie-feature matrices
        # i.e. the decomposition matrices (R = PQtranspose)
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_movies, self.K))

        # initializing the bias / baseline terms
        self.b_u = np.zeros(self.num_users)
        self.b_m = np.zeros(self.num_movies)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_movies)
            if self.R[i, j] > 0
            ]

        # stochastic gradient descent for given number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            rmse = self.rmse()
            training_process.append((i, rmse))
            if (i+1) % 20 == 0:
                print('Iteration: %d ; error = %.4f' % (i+1, rmse))
        return training_process

    # computing total root mean squared error (RMSE)
    def rmse(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    # stochastic gradient descent to get optimized P and Q
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_rating(i, j)
            e = (r-prediction)

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_m[j] += self.alpha * (e - self.beta * self.b_m[j])

    # get rating for user i and movie j
    def get_rating(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_m[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        full_matrix = self.b + self.b_u[:, np.newaxis] + self.b_m[np.newaxis:, ] + self.P.dot(self.Q.T)
        return full_matrix

# basic structure from here:
# https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/
