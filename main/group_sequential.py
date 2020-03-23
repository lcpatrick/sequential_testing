import numpy as np

class StandardGroupSequential(object):
    """
    This class provides the appropriate critical Z-value to be used for each z-test, where K such
    z-tests are planned to be conducted. The methodology assumes that each z-test employs the same
    critical z-value.
    """

    def __init__(
        self,
        num_tests,
        alpha=0.05,
        simulation_size=10000
    ):
        """
        num_tests: number of tests planned, K
        alpha: maximum total Type I error committed by all K tests, defaults to 0.05
        simulation_size: the number of random draws used in simulating the z-statistics
        """
        super().__init__()
        self.num_tests = num_tests
        self.alpha = alpha
        self.simulation_size = simulation_size

    def generate_covariance_matrix(self):
        sigma = np.ones((self.num_tests, self.num_tests))
        for j in range(self.num_tests):
            for k in range(self.num_tests):
                sigma[j,k] = np.sqrt(min(j+1, k+1) / max(j+1, k+1))
        self.sigma = sigma

    def compute_critical_z_value(self):
        """
        Returns the p_value that can be used (and is the same for) each of the K tests.
        Calculation is done via simulations. The steps are as follows.
        1. For each day, draw M multivariate normal random variables with covariance structure
        Sigma. These random draws can be thought of as simulting the z-statistics from each of the
        K z-tests. This step yields a Z matrix of shape M by K.
        2. Compute maximum absolute z-statistics. This step yields a Z_abs matrix of shape M by 1.
        3. Compute (1-alpha)-percential of Z_abs as the critical Z value Z_c.
        """

        self.generate_covariance_matrix()
        mean = np.zeros((self.num_tests,))
        Z = np.random.multivariate_normal(mean, self.sigma, self.simulation_size)
        Z_abs = np.max(np.abs(Z), axis=1).reshape((-1,1))
        Z_c = np.percentile(Z_abs, 100 * (1 - self.alpha))
        return Z_c