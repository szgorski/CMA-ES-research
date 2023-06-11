from Function import Function
from Strategy import Strategy
import numpy as np


class CMAES(Strategy):
    def __init__(self, function: Function, max_iterations: int):
        # problem initialization
        super().__init__(function, max_iterations)
        self.dim = self.func.dim

        # results
        self.best_x = None
        self.best_value = None

    def calculate(self):
        # a vector of means for each dimension (initialized with random values from [0.0, 1.0))
        mean = np.random.random(self.dim)

        sigma = 1  # neutral element of multiplication
        lamb = 4 + int(3 * np.log(self.dim))          # values sourced from ...
        mu = int(lamb / 2)  # TODO fn source

        # weights assigned from the highest-ranked to less important
        w = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)])  # TODO fn source
        w /= sum(w)                                     # normalization to 1
        mu_eff = 1 / np.sum(np.power(w, 2) for w in w)  # mu efficiency (const)

        # TODO fn source
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)                 # C path forget ratio
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)                                       # sigma path forget ratio
        c_1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)                                             # RANK-1 update ratio
        c_mu = min([1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2) ** 2 + mu_eff)])  # RANK-MU update ratio
        c_last = 1 - c_1 - c_mu                                                             # C matrix succession ratio

        path_c = np.zeros(self.dim)         # C evolution path  (accumulation of historical values of C)
        path_sigma = np.zeros(self.dim)     # sigma evolution path (accumulation of historical values of sigma)
        c_matrix = np.eye(self.dim)         # covariance matrix C
        
        self.best_x = 0
        self.best_value = np.inf
        for it in range(self.max_iter):
            # calculate M = square root of C (https://stackoverflow.com/questions/61262772)
            eigenvalues, eigenvectors = np.linalg.eigh(c_matrix)
            m_matrix = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.transpose()

            # create lambda new samples
            z = np.zeros((lamb, self.dim))
            d = np.zeros((lamb, self.dim))
            x = np.zeros((lamb, self.dim))

            for i in range(lamb):
                z[i] = np.random.normal(0, 1, self.dim)   # generation of plain samples from N(0, 1)
                d[i] = np.dot(m_matrix, z[i])             # placement of samples in the space with respect to matrix M
                x[i] = mean + sigma * d[i]                # dispersion of samples with respect to sigma

            # evaluate samples and update mean
            score = np.zeros(lamb)
            for i in range(lamb):
                score[i] = self.evaluate(x[i])

            # sort samples by score
            order = np.argsort(score)
            if score[order[0]] < self.best_value:
                self.best_x = x[order[0]]
                self.best_value = score[order[0]]

            # find and apply the mean of mu best samples
            mean = mean + sigma * np.sum(w[i] * d[order[i]] for i in range(mu))

            # evolution path update
            p_sqrt = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
            weighted_d = np.sum(w[i] * d[order[i]] for i in range(mu))
            weighted_z = np.sum(w[i] * z[order[i]] for i in range(mu))

            path_c = (1 - c_c) * path_c + p_sqrt * weighted_d
            path_sigma = (1 - c_sigma) * path_sigma + p_sqrt * weighted_z

            # RANK-1
            p_matrix = path_c.reshape(self.dim, 1)
            p_matrix_t = path_c.reshape(1, self.dim)
            rank_1 = c_1 * np.dot(p_matrix, p_matrix_t)

            # RANK-MU
            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(mu):
                d_matrix = d[order[i]].reshape(self.dim, 1)
                d_matrix_t = d[order[i]].reshape(1, self.dim)
                rank_mu += c_mu * w[i] * np.dot(d_matrix, d_matrix_t)

            # covariance matrix update
            last_c = c_last * c_matrix
            c_matrix = last_c + rank_1 + rank_mu

            # step-size update TODO fn source
            sigma *= np.exp((c_sigma / 2) * (np.sum(np.power(x, 2) for x in path_sigma) / self.dim - 1))
            print(it + 1, self.best_value)

    def get_results(self):
        return self.best_value, self.best_x

    def evaluate(self, value):
        return self.func.evaluate(value)
