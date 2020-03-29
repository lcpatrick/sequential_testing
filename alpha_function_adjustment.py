import numpy as np
from scipy import stats

class AlphaFunctionAdjustment(object):
    """
    Provide adjustments to critical z values and p values so that overall Type I error is respected, and Type I error follows a specified path, if given.
    """

    def __init__(
        self,
        setting,
        alpha,
        num_tests,
        alpha_function=None,
        mean=None,
        sigma_m=None,
        z_simulation_runs=10000,
        z_grid = np.linspace(1.96, 5.00, 500).reshape((1, -1))
    ):
        """
        -- setting: 'sequential' or 'mht' (multiple hypothesis testing)
        -- alpha: total Type I error
        -- num_tests: total number of tests
        -- alpha_function: the schedule of how much total alpha can be spent, as a increasing numpy array
        -- mean: mean of the metrics, applicable only for 'mht' setting
        -- sigma_m: covariance of the metrics, applicable only for the 'mht' setting
        -- z_simulation_runs: number of simulations of z statistics in computing critical values
        -- z_grid: grid of z values to perform search over
        """
        self.setting = setting
        self.alpha = alpha
        self.num_tests = num_tests
        self.alpha_function = alpha_function
        self.mean = mean
        self.sigma_m = sigma_m
        self.z_simulation_runs = z_simulation_runs
        self.z_grid = z_grid
        # check that num_tests and alpha_function are in agreement
        if np.any(alpha_function):
            if num_tests != alpha_function.shape[0]:
                raise ValueError("Number of tests is inconsistent with dimension of alpha function.")
        if np.any(alpha_function) and alpha:
            if alpha != alpha_function[-1]:
                raise ValueError("Total Type I error is inconsistent with alpha function.")
        # check that if setting 'mht' is used, then mean and sigma_m must be specified and are consistent
        # in dimensions
        if setting == 'mht':
            if not np.any(mean) or not np.any(sigma_m):
                raise ValueError("Metric means and covariances must be provided for multiple hypothesis testing adjustments.")
            if np.any(mean.shape[0] != sigma_m.shape[0]):
                raise ValueError("Dimensions of metric means and variances are inconsistent.")
            if np.any(alpha_function):
                if alpha_function.shape[0] != mean.shape[0]:
                    raise ValueError("Number of metrics is inconsistent with dimension of alpha function.")

    def compute_all_variable_critical_values(self):
        if not np.all(self.alpha_function):
            raise ValueError("Alpha function must be provided.")
        critical_values = []
        for _ in range(len(self.alpha_function)):
            z = self._compute_next_variable_critical_value(critical_values)
            critical_values.append(z)
        return critical_values

    def _compute_next_variable_critical_value(self, previous_critical_values):
        # given a set of critical values for previous tests, calculate the right critical value
        # for the next test to satisfy the alpha function
        reject = self._simulate_rejection(previous_critical_values)
        k = len(previous_critical_values)
        freq = np.mean(reject, axis=0)
        index = np.argmin(np.abs(freq - self.alpha_function[k]))
        return self.z_grid[0, index]

    def _simulate_rejection(self, previous_critical_values):
        """
        Returns rejection decisions as a function of the grid of critical Z-values at test k.
        Arguments:
        -- previous_Z_values: a list of computed critical Z-values for tests 1 to j, can be empty for the very first test
        """
        Z = self._simulate_Z_statistics()
        Z_abs = np.abs(Z)
        # keeps track of whether there's any rejection at any test from 1 to k
        accept = np.array([True]).reshape((-1, 1))
        j = len(previous_critical_values)
        if j > 0:
            for i in range(j):
            # compare the simulated Z statistics against the pre-computed critical values for previous tests
            # from 1 to j
                z = previous_critical_values[i]
                compare = Z_abs[:, i:i+1] < z
                accept = np.multiply(accept, compare)
        # for test k, compare the Z-statistic against a grid
        compare_k = Z_abs[:, j:j+1] < self.z_grid
        # combine accept decision across all k tests
        accept = np.multiply(accept, compare_k)
        # infer rejections
        reject = 1 - accept
        return reject

    def compute_constant_critical_value(self):
        if not self.alpha:
            raise ValueError("Overall Type I error must be specified.")
        # compute the constant critical value that can be used across tests to obtain Type I error alpha
        Z = self._simulate_Z_statistics()
        Z_abs = np.max(np.abs(Z), axis=1).reshape((-1,1))
        critical_z = np.percentile(Z_abs, 100 * (1 - self.alpha))
        p_value = 2 * (1 - stats.norm.cdf(critical_z, loc=0, scale=1))
        return critical_z, p_value

    def _simulate_Z_statistics(self):
        # simulate z statistics for all the tests
        sigma_z = self._compute_cor_z()
        mean_z = np.zeros((self.num_tests,))
        Z = np.random.multivariate_normal(mean_z, sigma_z, self.z_simulation_runs)
        return Z

    def _compute_cor_z(self):
        if self.setting == 'sequential':
            return self._compute_cor_z_sequential()
        elif self.setting == 'mht':
            return self._compute_cor_z_mht()

    def _compute_cor_z_mht(self):
        # returns the correlation matrix of z statistics for the multiple hypothesis testing setting
        sigma_z = np.ones((self.num_tests, self.num_tests))
        for i in range(self.num_tests):
            for j in range(self.num_tests):
                sigma_z[i, j] = self.sigma_m[i, j] / np.sqrt(self.sigma_m[i, i] * self.sigma_m[j, j])
        return sigma_z

    def _compute_cor_z_sequential(self):
        # returns the correlation matrix of z statistics for the sequential testing setting
        sigma_z = np.ones((self.num_tests, self.num_tests))
        for j in range(self.num_tests):
            for k in range(self.num_tests):
                sigma_z[j,k] = np.sqrt(min(j+1, k+1) / max(j+1, k+1))
        return sigma_z

    def compute_alpha(self, critical_values):
        if len(critical_values) == 1:
            critical_value = critical_values[0]
            return self._compute_alpha_constant_critical_value(critical_value)
        return self._compute_alpha_variable_critical_values()

    def _compute_alpha_constant_critical_value(self, critical_value):
        previous_critical_values = [critical_value]
        alphas = []
        for i in range(self.num_tests):
            rejection_rate = self._simulate_rejection(self, previous_critical_values).mean()


