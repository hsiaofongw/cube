import unittest
import numpy as np
from helper import Helper
from math import cos, dist
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
    
    def test_projection(self):

        n_dims = 3
        vec_b = np.random.rand(n_dims)
        vec_a = np.random.rand(n_dims)

        vec_c = Helper.projection(vec_b, vec_a)

        lhs = Helper.cosine(vec_b, vec_c)
        rhs = 1.0
        eps = 1e-6
        self.assertAlmostEqual(lhs, rhs, delta=eps)
    
    def test_vertical_line(self):

        n_dims = 3
        point_c = np.random.rand(n_dims)
        point_a = np.random.rand(n_dims)
        point_b = np.random.rand(n_dims)

        vec_ca = point_a - point_c
        vec_db = Helper.vertical_line(point_c, point_a, point_b)
        lhs = Helper.cosine(vec_db, vec_ca)
        rhs = 0.0
        eps = 1e-6
        self.assertAlmostEqual(lhs, rhs, delta=eps)
    
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
    
    def test_check_intersect(self):
 
        n_samples = 100
        scale_factor = 100
        points_o = np.random.rand(n_samples, 3) * scale_factor
        points_a = np.random.rand(n_samples, 3) * scale_factor
        points_b = np.random.rand(n_samples, 3) * scale_factor
        points_c = np.random.rand(n_samples, 3) * scale_factor
        vecs_ba = points_b - points_a
        vecs_ca = points_c - points_a

        true_xs = np.random.rand(n_samples, 1)
        true_ys = np.random.rand(n_samples, 1)
        points_p = points_a + true_xs * vecs_ba + true_ys * vecs_ca
        distances = np.linalg.norm(points_p - points_o, axis=1).reshape(n_samples, 1)
        portions = np.random.rand(n_samples, 1)
        vecs_v = points_p - points_o
        vecs_v_norm = np.linalg.norm(vecs_v, axis=1).reshape(vecs_v.shape[0], 1)
        vecs_v = vecs_v / vecs_v_norm
        v_lengths = distances * portions
        vecs_v = v_lengths * vecs_v
        true_zs = distances / v_lengths

        mat_Aeq = np.zeros(shape=(1, 2,), dtype=np.float)
        mat_Beq = np.zeros(shape=(2, 1,), dtype=np.float)

        mat_Aleq = np.array([
            [1, 1],
            [-1, 0],
            [0, -1]
        ])
        mat_Bleq = np.array([
            [1],
            [0],
            [0]
        ])

        computed_xs = np.zeros_like(true_xs)
        computed_ys = np.zeros_like(true_ys)
        computed_zs = np.zeros_like(true_zs)
        for i in range(n_samples):
            point_o = points_o[i, :]
            v = vecs_v[i, :]
            point_a = points_a[i, :]
            point_b = points_b[i, :]
            point_c = points_c[i, :]
            check, answer = Helper.check_intersect(
                point_o, 
                v, 
                point_a, 
                point_b, 
                point_c, 
                mat_Aeq, 
                mat_Beq, 
                mat_Aleq, 
                mat_Bleq
            )
            computed_zs[i, 0] = answer[0, 0]
            computed_xs[i, 0] = answer[1, 0]
            computed_ys[i, 0] = answer[2, 0]
        
        true_values = np.concatenate((true_zs, true_xs, true_ys,), axis=1)
        computed_values = np.concatenate((computed_zs, computed_xs, computed_ys,), axis=1)

        eps = 1e-6
        errors = np.abs(true_values - computed_values)
        max_error = np.max(errors)
        self.assertAlmostEqual(max_error, 0.0, delta=eps)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored', '-v'])
        