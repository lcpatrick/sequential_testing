import numpy as np
from scipy import stats

class MHT(object):
    def __init__(self, alpha, mean, sigma_m, sims=100000, num_units=1000):
        super().__init__()
        self.alpha = alpha # Type I error
        self.mean = mean # mean of the metrics, numpy array
        self.sigma_m = sigma_m # covariance matrix of the metrics, numpy array
        self.sims = sims # number of simulations
        self.num_units = num_units # number of simulated units in each variant

    def _get_sigma_z(self):
        # compute the covariance matrix of the z_statistics
        K = self.sigma_m.shape[0] # number of metrics
        sigma_z = np.ones((K, K))
        for i in range(K):
            for j in range(K):
                sigma_z[i, j] = sigma_m[i, j] / np.sqrt(sigma_m[i, i] * sigma_m[j, j])
        return sigma_z

    def compute_constant_p_value(self):
        sigma_z = self._get_sigma_z()
        K = sigma_z.shape[0] # number of metrics
        Z_statistics = np.zeros((self.sims, K))
        z_mean = np.zeros((K))
        Z = np.random.multivariate_normal(z_mean, sigma_z, self.sims)
        Z_abs = np.max(np.abs(Z), axis=1)
        z_c = np.percentile(Z_abs, 100 * (1 - self.alpha))
        p_value = 2 * (1 - stats.norm.cdf(z_c, loc=0, scale=1))
        print("adjusted p_value is {}".format(p_value))
        print("adjusted critical z value is {}".format(z_c))
        return (z_c, p_value)

    def compute_overall_reject_freq(self, critical_value):
        """
        computes the overall Type I error, given a critical value
        critical_value: the value to determine rejection
        """
        K = sigma_m.shape[0] # number of metrics
        N = self.num_units # number of units in each variant
        Z = np.zeros((self.sims, K))
        for sim in range(self.sims):
            # simulate data, and compute z_statistics for each of the K metrics
            data_A = np.random.multivariate_normal(mean, self.sigma_m, N)
            data_B = np.random.multivariate_normal(mean, self.sigma_m, N)
            # compute z_statistic for test on each metric, and store
            for metric in range(K):
                delta = data_A[:, metric].mean() - data_B[:, metric].mean()
                std = np.sqrt(
                    data_A[:, metric].var() / N +
                    data_A[:, metric].var() / N)
                z = delta / std
                Z[sim, metric] = z

        # count frequency of rejecting null for at least one metric
        reject_freq = (np.max(np.abs(Z), axis=1) > critical_value).mean()
        print("rejection frequency is {}".format(reject_freq))

    def compute_alpha_expenditure(self, critical_value):
        """
        computes how much of Type I error each test incurs
        """
        sigma_z = self._get_sigma_z()
        K = sigma_z.shape[0] # number of metrics
        alphas = np.zeros((K,)) # store of Type I error incurred
        Z_statistics = np.zeros((self.sims, K))
        z_mean = np.zeros((K))
        Z = np.random.multivariate_normal(z_mean, sigma_z, self.sims)
        for i in range(K):
            Z_abs = np.max(np.abs(Z[:, :i+1]), axis=1)
            alpha = (Z_abs > critical_value).mean()
            alphas[i] = alpha
        return alphas


