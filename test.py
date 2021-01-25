import unittest
import numpy as np
from helper import Helper
from math import cos
from scipy.spatial.distance import cosine as cosine_distance

class TestHelperMethods(unittest.TestCase):

    def test_cosine(self):

        vec_a = np.random.rand(3)
        vec_b = np.random.rand(3)

        lhs = Helper.cosine(vec_a, vec_b)
        rhs = 1 - cosine_distance(vec_a, vec_b)
        eps = 1e-6
        self.assertAlmostEqual(lhs, rhs, delta=eps)
    
    def test_cosine_many(self):
        n_samples = 10
        n_dims_of_vector = 3
        lhs_points: np.ndarray = np.random.rand(n_samples, n_dims_of_vector)
        rhs_points: np.ndarray = np.random.rand(lhs_points.shape[0], lhs_points.shape[1])

        cosines = Helper.cosine_many(lhs_points, rhs_points)
        lhs: np.ndarray = cosines
        rhs: np.ndarray = np.zeros_like(lhs)

        for i in range(rhs.shape[0]):
            lhs_point = lhs_points[i, :]
            rhs_point = rhs_points[i, :]
            rhs[i] = Helper.cosine(lhs_point, rhs_point)

        differences = lhs - rhs
        errors = np.abs(differences)
        total_errors = np.sum(errors)
        mean_error = total_errors/n_samples

        eps = 1e-6
        self.assertAlmostEqual(mean_error, 0, delta=eps)
    
    def test_rotate_one(self):
        point_p = np.random.rand(3)
        point_c = np.random.rand(3)
        point_t = np.random.rand(3)
        theta = np.random.rand()
        
        vec_dp1, point_p1 = Helper.rotate_one(point_c, point_t, point_p, theta)
        point_d = point_p1 - vec_dp1
        vec_dp = point_p - point_d

        lhs = Helper.cosine(vec_dp, vec_dp1)
        rhs = cos(theta)
        eps = 1e-6
        self.assertTrue(abs(lhs-rhs) < eps)
        
        vec_ct = point_t - point_c
        lhs = Helper.cosine(vec_dp, vec_ct)
        rhs = Helper.cosine(vec_dp1, vec_ct)
        self.assertTrue(abs(lhs-rhs) < eps)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored', '-v'])
        