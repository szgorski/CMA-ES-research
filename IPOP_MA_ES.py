from Strategy import *
import numpy as np

DELTA = 2
MAX_POPULATION = 10000

# termination criteria
TOLX = 1e-12
TOLXUP = 1e4
TOLFUN = 1e-12
TOLCOND = 1e14
EPSILON = 1e-8


class IPOP_MAES(Strategy):
    def __init__(self,
                 function: Callable,
                 x_initial: Any,
                 max_iterations: int,
                 limit_evaluations: bool,
                 seed: int | None = None):
        super().__init__(function, x_initial, max_iterations, limit_evaluations, seed)

    def stop_condition(self, m_matrix, it, best_iter_score, worst_iter_score, sigma, path_sigma):

        eigenvalues, eigenvectors = np.linalg.eigh(m_matrix)
        if abs(best_iter_score - worst_iter_score) < TOLFUN:
            return True

        if sigma * np.max(eigenvectors) > TOLXUP:
            return True

        if np.linalg.cond(m_matrix) > TOLCOND:
            return True

        diag = np.diag(m_matrix)
        if np.all(sigma * diag < TOLX) and np.all(sigma * path_sigma < TOLX):
            return True

        if np.any(abs(0.2 * sigma * diag) < TOLFUN):
            return True

        i = it % self.dim
        if np.all(abs(0.1 * sigma * eigenvalues[i] * eigenvectors[:, i]) < EPSILON):
            return True

        if np.isnan(np.max(m_matrix)):
            return True

        return False

    def calculate(self):
        # a vector of means for each dimension (initialized with given values)
        mean = self.x_init

        eval_count = 0
        eval_left = self.max_eval
        sigma = 1                                                               # neutral element of multiplication
        lamb = 4 + int(3 * np.log(self.dim))
        mu = int(lamb / 2)  # TODO fn

        if self.max_iter is None:
            self.max_iter = np.ceil(self.max_eval / lamb)

        # weights assigned from the highest-ranked to less important
        w = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)])
        w /= sum(w)                                                             # normalization to 1
        mu_eff = 1 / np.sum(np.power(w, 2) for w in w)                          # mu efficiency (const)

        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)                        # sigma path forget ratio
        c_1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)                              # RANK-1 update ratio
        c_mu = min([1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) /
                    ((self.dim + 2) ** 2 + mu_eff)])                            # RANK-MU update ratio

        # sigma evolution path (accumulation of historical values of sigma)
        path_sigma = np.zeros(self.dim)
        m_matrix = np.eye(self.dim)  # matrix M

        self.best_x = 0
        self.best_value = np.inf

        it = 0
        while it < self.max_iter:
            # create lambda new samples
            z = np.zeros((lamb, self.dim))
            d = np.zeros((lamb, self.dim))
            x = np.zeros((lamb, self.dim))

            for i in range(lamb):
                z[i] = self.rand.normal(0, 1, self.dim)     # generation of plain samples from N(0, 1)
                d[i] = np.dot(m_matrix, z[i])               # placement of samples in the space with respect to matrix M
                x[i] = mean + sigma * d[i]                  # dispersion of samples with respect to sigma

            # evaluate samples and update mean
            score = np.zeros(lamb)
            for i in range(lamb):
                score[i] = self.func(x[i])
                eval_count += 1

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
            rank_1 = c_1 / 2 * (np.dot(p_matrix, p_matrix_t) - np.eye(self.dim, self.dim))

            # RANK-MU
            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(mu):
                z_matrix = z[order[i]].reshape(self.dim, 1)
                z_matrix_t = z[order[i]].reshape(1, self.dim)
                rank_mu += c_mu * w[i] * np.dot(z_matrix, z_matrix_t)
            rank_mu = c_mu / 2 * (rank_mu - np.eye(self.dim, self.dim))

            # matrix M update
            m_matrix = np.dot(m_matrix, np.eye(self.dim, self.dim) + rank_1 + rank_mu)
            sigma *= np.exp((c_sigma / 2) * (np.sum(np.power(x, 2) for x in path_sigma) / self.dim - 1))
            it += 1

            if self.stop_condition(m_matrix, it, order[0], order[-1], sigma, path_sigma) and lamb < MAX_POPULATION:
                # a vector of means for each dimension (initialized randomly in the search space)
                mean = self.lb + (self.rand.random(self.dim) * (self.ub - self.lb))

                sigma = 1                                                       # neutral element of multiplication
                lamb *= DELTA
                mu = int(lamb / 2)

                if self.limit_eval:
                    eval_left = eval_left - eval_count
                    eval_count = 0
                    self.max_iter = np.ceil(eval_left / lamb)
                    it = 0

                # weights assigned from the highest-ranked to less important
                w = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)])
                w /= sum(w)                                                     # normalization to 1
                mu_eff = 1 / np.sum(np.power(w, 2) for w in w)                  # mu efficiency (const)

                c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)                # sigma path forget ratio
                c_1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)                      # RANK-1 update ratio
                c_mu = min([1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) /
                            ((self.dim + 2) ** 2 + mu_eff)])                    # RANK-MU update ratio

                # sigma evolution path (accumulation of historical values of sigma)
                path_sigma = np.zeros(self.dim)
                m_matrix = np.eye(self.dim)                                    # matrix M
