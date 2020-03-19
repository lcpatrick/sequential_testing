import numpy as np

class GroupSequential(object):
    """
    This class provides the appropriate p_value for each peek at the experiment results. It is
    assumed, per group sequential testing methodology, that (1) the experiment is set to run for a
    fixed number of N days (or weeks), (2) the number of peeks at the data is fixed to be a number
    K, and (3) the peeks take place at regular, pre-defined intervals each of length N/K days.
    For example, an experiment can be set to run for 4 weeks, and the experimenter can look at the
    results 4 times, once at the end of each week (to call the experiment) - so N = 4 (weeks) and
    K = 4 (peeks).

    (The adjustment of p_value is necessary to prevent committing large Type I error in the presence
    of repeated testing/peeking.)
    """

    def __init__(
        self,
        num_days,
        num_tests,
        type='constant_p_value',
        alpha=0.05,
        simulation_size=10000
    ):
        """
        num_days: number of days the experiment is set to run, N
        num_tests: number of tests planned, K
        type: controls how p_values are determined
            'constant_p_values' (default):  p_values are the same for each of the K sequential tests
            'constant_type_I_increase': p_values set to incur the same amount of additional type I
            error for each test
        alpha: maximum total Type I error committed by all K tests, defaults to 0.05
        """
        super().__init__()
        self.num_days = num_days
        self.num_tests = num_tests
        self.alpha = alpha
        self.simulation_size = simulation_size

    def compute_constant_p_value(self):
        """
        Returns the p_value that can be used (and is the same for) each of the K tests.
        Calculation is done via simulations. The steps are as follows.
        1. For each day, draw M standard normal random outcomes. These outcomes can be thought of
        as representing the null hypothesis of no treatment effect. The average of these outcomes
        are normal, with mean zero.
        2. Compute the z-test statistic at the pre-defined intervals. Call them Z_k for k from 1
        to K. Note that each Z_k follows an asymptotic N(0, 1) distribution.
        3. Compute critical value Z such that
        P(max_{k=1,...K} (Z_k) > Z) = alpha.
        That is, find the critical value Z such that the probability that any single test yields
        a test statistic more than Z is exactly alpha.
        """
        X = np.random.normal(self.num_days * self.simulation_size, 0, 1)

