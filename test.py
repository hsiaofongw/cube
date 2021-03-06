import unittest
import numpy as np
from scipy.sparse.construct import block_diag
from helper import Helper
from math import cos, pi
from scipy.spatial.distance import cosine as cosine_distance
from cube import Cube

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
        direction = np.random.rand(3)
        center = np.random.rand(3)
        theta = np.random.rand()

        point_p1 = Helper.rotate_one(point_p, direction, center, theta)

        vec_cp1 = point_p1 - center
        vec_cp = point_p - center
        projection = direction * np.inner(direction, vec_cp) / np.inner(direction, direction)
        vec_dp1 = vec_cp1 - projection
        vec_dp = vec_cp - projection

        tolerance = 1e-6
        lhs = Helper.cosine(vec_dp1, vec_dp)
        rhs = cos(theta)
        self.assertAlmostEqual(lhs, rhs, delta=tolerance)

        tolerance = 1e-6
        lhs = Helper.cosine(vec_dp1, direction)
        rhs = 0.0
        self.assertAlmostEqual(lhs, rhs, delta=tolerance)

        tolerance = 1e-6
        lhs = Helper.cosine(vec_dp, direction)
        rhs = 0.0
        self.assertAlmostEqual(lhs, rhs, delta=tolerance)

    def test_rotate_many(self):

        n_samples = 100
        points_p = np.random.rand(n_samples, 3)
        direction = np.random.rand(3)
        center = np.random.rand(3)
        theta = np.random.rand()

        points_p1 = Helper.rotate_many(points_p, direction, center, theta)
        vecs_cp = points_p - center
        vecs_cp1 = points_p1 - center
        inner_ab = np.sum((direction.reshape(1,3)) * vecs_cp, axis=1)
        inner_aa = np.inner(direction, direction)
        projections = direction.reshape(1, 3) * (inner_ab / inner_aa).reshape((inner_ab.shape[0], 1))

        vecs_dp = vecs_cp - projections
        vecs_dp1 = vecs_cp1 - projections

        tolerance = 1e-6
        lhs = np.max(np.abs(np.diagonal(Helper.cosine_many_to_many(vecs_dp, vecs_dp1)) - cos(theta)))
        rhs = 0.0
        self.assertAlmostEqual(lhs, rhs, delta=tolerance)

        tolerance = 1e-6
        lhs = np.max(np.abs(
            Helper.cosine_many_to_many(
                direction.reshape(1, 3),
                vecs_dp
            ) - 0
        ))
        rhs = 0.0
        self.assertAlmostEqual(lhs, rhs, delta=tolerance)

        tolerance = 1e-6
        np.max(np.abs(
            Helper.cosine_many_to_many(
                direction.reshape(1, 3),
                vecs_dp1
            ) - 0
        ))
        rhs = 0.0
        self.assertAlmostEqual(lhs, rhs, delta=tolerance)

    def test_find_intersect(self):
 
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

        computed_xs = np.zeros_like(true_xs)
        computed_ys = np.zeros_like(true_ys)
        computed_zs = np.zeros_like(true_zs)
        for i in range(n_samples):
            point_o = points_o[i, :]
            v = vecs_v[i, :]
            point_a = points_a[i, :]
            point_b = points_b[i, :]
            point_c = points_c[i, :]
            answer = Helper.find_intersect(
                point_o, 
                v, 
                point_a, 
                point_b, 
                point_c
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
    
    def test_cosine_many_to_many(self):

        xs = np.random.rand(100, 3)
        ys = np.random.rand(120, 3)
        cosines1 = Helper.cosine_many_to_many(xs, ys)

        cosines2 = np.zeros_like(cosines1)
        for i in range(cosines2.shape[0]):
            for j in range(cosines2.shape[1]):
                cosines2[i, j] = 1 - cosine_distance(xs[i, :], ys[j, :])
        
        errors = cosines1 - cosines2
        max_error = np.max(np.abs(errors))
        tolerance = 1e-6
        self.assertAlmostEqual(max_error, 0.0, delta=tolerance)
    
    def test_make_diagonal_matrix(self):

        mat_A1 = np.random.rand(3, 4)
        mat_A2 = np.random.rand(3, 4)
        mat_A3 = np.random.rand(3, 4)
        mat_A4 = np.random.rand(3, 4)
        data = np.concatenate((mat_A1, mat_A2, mat_A3, mat_A4,), axis=0)
        n_rows = 3

        s = Helper.make_block_diagonal(data, n_rows)
        block_diagonal = s.toarray()

        mat_B1 = block_diagonal[0:3, 0:4]
        mat_B2 = block_diagonal[3:6, 4:8]
        mat_B3 = block_diagonal[6:9, 8:12]
        mat_B4 = block_diagonal[9:12, 12:16]

        lhs = data
        rhs = np.concatenate((mat_B1, mat_B2, mat_B3, mat_B4,), axis=0)
        errors = lhs - rhs
        max_error = np.max(np.abs(errors))
        tolerance = 1e-6
        self.assertAlmostEqual(max_error, 0.0, delta=tolerance)

class TestCubeMethods(unittest.TestCase):

    def test_rotate(self):

        cube_center = np.array([0, 0, 0])
        edge_length = 1
        cube = Cube(
            cube_center,
            edge_length
        )

        edge_lengths_before = cube.get_edge_lengths()

        rotate_axis = np.array([0, 0, 1])
        radian = pi/6
        cube.rotate(rotate_axis, cube.cube_center, radian)

        edge_lengths_after = cube.get_edge_lengths()

        max_error = np.max(np.abs(edge_lengths_after - edge_lengths_before))

        tolerance = 1e-6
        self.assertAlmostEqual(max_error, 0.0, delta=tolerance)
    
    def test_scale(self):

        cube_center = np.array([0, 0, 0])
        edge_length = 1
        cube = Cube(
            cube_center,
            edge_length
        )

        edge_lengths_before = cube.get_edge_lengths()

        factor = np.random.rand()
        cube.scale(factor)

        edge_lengths_after = cube.get_edge_lengths()

        tolerance = 1e-6
        lhs = edge_lengths_before * factor
        rhs = edge_lengths_after
        max_error = np.max(np.abs(lhs - rhs))
        self.assertAlmostEqual(max_error, 0.0, delta=tolerance)

    
    def test_move(self):

        cube_center = np.array([0, 0, 0])
        edge_length = 1
        cube = Cube(
            cube_center,
            edge_length
        )

        edge_lengths_before = cube.get_edge_lengths()

        replacement = np.random.rand(3)
        cube.move(replacement)

        edge_lengths_after = cube.get_edge_lengths()

        tolerance = 1e-6
        lhs = edge_lengths_before
        rhs = edge_lengths_after
        max_error = np.max(np.abs(lhs - rhs))
        self.assertAlmostEqual(max_error, 0.0, delta=tolerance)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored', '-v'])
        