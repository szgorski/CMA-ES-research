from Strategy import *
import numpy as np

DELTA = 10
MAX_POPULATION = 1000


class MAES(Strategy):
    def __init__(self,
                 function: Callable,
                 x_initial: Any,
                 max_iterations: int,
                 limit_evaluations: bool,
                 seed: int | None = None):
        super().__init__(function, x_initial, max_iterations, limit_evaluations, seed)

    def calculate(self):
        # a vector of means for each dimension (initialized with given values)
        mean = self.x_init

        sigma = 1  # neutral element of multiplication
        lamb = 4 + int(3 * np.log(self.dim))  # values sourced from ...
        mu = int(lamb / 2)  # TODO fn

        if self.max_iter is False:
            self.max_iter = np.ceil(self.max_eval / lamb)

        # weights assigned from the highest-ranked to less important
        w = np.array([np.log(mu + 0.5) - np.log(i + 1)
                     for i in range(mu)])  # TODO fn
        # normalization to 1
        w /= sum(w)
        mu_eff = 1 / np.sum(np.power(w, 2)
                            for w in w)      # mu efficiency (const)

        # sigma path forget ratio
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        # RANK-1 update ratio
        c_1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)
        c_mu = min([1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) /
                   ((self.dim + 2) ** 2 + mu_eff)])  # RANK-MU update ratio

        # sigma evolution path (accumulation of historical values of sigma)
        path_sigma = np.zeros(self.dim)
        m_matrix = np.eye(self.dim)         # matrix M

        self.best_x = 0
        self.best_value = np.inf

        for it in range(self.max_iter):
            # create lambda new samples
            z = np.zeros((lamb, self.dim))
            d = np.zeros((lamb, self.dim))
            x = np.zeros((lamb, self.dim))

            for i in range(lamb):
                # generation of plain samples from N(0, 1)
                z[i] = self.rand.normal(0, 1, self.dim)
                # placement of samples in the space with respect to matrix M
                d[i] = np.dot(m_matrix, z[i])
                # dispersion of samples with respect to sigma
                x[i] = mean + sigma * d[i]

            # evaluate samples and update mean
            score = np.zeros(lamb)
            for i in range(lamb):
                score[i] = self.func(x[i])

            # sort samples by score
            order = np.argsort(score)
            if score[order[0]] < self.best_value:
                self.best_x = x[order[0]]
                self.best_value = score[order[0]]

            # find and apply the mean of mu best samples
            mean = mean + sigma * np.sum(w[i] * d[order[i]] for i in range(mu))

            # evolution path update
            p_sqrt = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
            weighted_z = np.sum(w[i] * z[order[i]] for i in range(mu))
            path_sigma = (1 - c_sigma) * path_sigma + p_sqrt * weighted_z

            # RANK-1
            p_matrix = path_sigma.reshape(self.dim, 1)
            p_matrix_t = path_sigma.reshape(1, self.dim)
            rank_1 = c_1 / 2 * \
                (np.dot(p_matrix, p_matrix_t) - np.eye(self.dim, self.dim))

            # RANK-MU
            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(mu):
                z_matrix = z[order[i]].reshape(self.dim, 1)
                z_matrix_t = z[order[i]].reshape(1, self.dim)
                rank_mu += c_mu * w[i] * np.dot(z_matrix, z_matrix_t)
            rank_mu = c_mu / 2 * (rank_mu - np.eye(self.dim, self.dim))

            # matrix M update
            m_matrix = np.dot(m_matrix, np.eye(
                self.dim, self.dim) + rank_1 + rank_mu)
            if it % DELTA == 0 and self.lamb < MAX_POPULATION:
                lamb += DELTA

            sigma *= np.exp((c_sigma / 2) * (np.sum(np.power(x, 2)
                            for x in path_sigma) / self.dim - 1))
