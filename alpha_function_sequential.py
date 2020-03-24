import numpy as np

class AlphaFunctionSequential():

    def __init__(
        self,
        alphas = np.linspace(0, 0.05, 11)[1:],
        simulation_size=1000000,
        z_grid = np.linspace(1.96, 5.00, 1000).reshape((1, -1))
    ):
        """
        num_tests: number of tests planned, K
        alphas: alpha-spending function
        simulation_size: the number of random draws used in simulating the z-statistics
        """
        super().__init__()
        self.alphas = alphas
        self.simulation_size = simulation_size
        self.z_grid = z_grid
        self.num_tests = len(alphas)

    def compute_all_critical_Z_values(self):
        critical_values = []
        for _ in range(len(self.alphas)):
            z = self.compute_critical_Z_value(critical_values)
            critical_values.append(z)
            print(z)
        return critical_values

    def compute_critical_Z_value(self, previous_Z_values):
        reject = self._compute_rejection(previous_Z_values)
        k = len(previous_Z_values)
        freq = np.mean(reject, axis=0)
        index = np.argmin(np.abs(freq - self.alphas[k]))
        return self.z_grid[0, index]

    def _compute_rejection(self, previous_Z_values):
        """
        Returns rejection decisions as a function of the grid of critical Z-values at test k.
        Arguments:
        -- Z: the simulated Z-statistics
        -- previous_Z_values: a list of computed critical Z-values for tests 1 to j, can be empty for the very first test
        -- z_grid: an numpy array of values of possible critical Z-values to be used for the next test, test k
        -- alphas: the alpha-spending function
        """
        Z = self._simulate_Z_statistics()
        # keeps track of whether there's any rejection at any test from 1 to k
        accept = np.array([True]).reshape((-1, 1))
        j = len(previous_Z_values)
        if j > 0:
            for i in range(j):
            # compare the simulated Z statistics against the pre-computed critical values for previous tests
            # from 1 to j
                z = previous_Z_values[i]
                compare = Z[:, i:i+1] < z
                accept = np.multiply(accept, compare)
        # for test k, compare the Z-statistic against a grid
        compare_k = Z[:, j:j+1] < self.z_grid
        # combine accept decision across all k tests
        accept = np.multiply(accept, compare_k)
        # infer rejections
        reject = 1 - accept
        return reject

    def _simulate_Z_statistics(self):
        sigma = self._generate_covariance_matrix()
        mean = np.zeros((self.num_tests,))
        Z = np.abs(np.random.multivariate_normal(mean, sigma, self.simulation_size))
        return Z

    def _generate_covariance_matrix(self):
        sigma = np.ones((self.num_tests, self.num_tests))
        for j in range(self.num_tests):
            for k in range(self.num_tests):
                sigma[j,k] = np.sqrt(min(j+1, k+1) / max(j+1, k+1))
        return sigma

