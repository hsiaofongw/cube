import numpy as np
from scipy.spatial.transform import Rotation as R
from math import sin, cos, floor
from scipy.spatial.distance import cosine as cosine_distance
from typing import Tuple
from scipy.sparse import csr_matrix

class Helper:
    
    # input:
    # x: 1-D array as a vector, y: 1-D array, both have equal length
    # output: float as their cosine
    @classmethod
    def cosine(cls, x: np.ndarray, y: np.ndarray) -> float:
        return 1 - cosine_distance(x, y)
    
    # input:
    # many_x: n by m array, many_y: n by m array, each row for a vector
    # output: 1-D array with n elements, output[i] = cosine(many_x[i, :], many_y[i, :])
    @classmethod
    def cosine_many(cls, many_x: np.ndarray, many_y: np.ndarray) -> np.ndarray:
        inner_prods = np.sum(many_x * many_y, axis = 1)
        x_norms = np.linalg.norm(many_x, axis=1)
        y_norms = np.linalg.norm(many_y, axis=1)
        cosines = inner_prods / (x_norms * y_norms)
        return cosines
    
    # input:
    # xs: n by d matrix, each row is vector of length d
    # ys: m by d matrix, each row is vector of length d
    # output: a n by m matrix, is [i, j] element equals cosine(xs[i, :], ys[j, :])
    @classmethod
    def cosine_many_to_many(cls, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        xs = xs / (np.linalg.norm(xs, axis=1).reshape(xs.shape[0],1))
        ys = ys / (np.linalg.norm(ys, axis=1).reshape(ys.shape[0],1))
        cosines = xs @ (ys.T)
        return cosines
    
    # input:
    # vec_b: a vector
    # vec_a: a vector
    # output: project of vec_a in vec_b
    @classmethod
    def projection(cls, vec_b: np.ndarray, vec_a: np.ndarray) -> np.ndarray:
        return vec_b * (np.inner(vec_a, vec_b)/np.sum(vec_b * vec_b))
    
    # input:
    # point_c: a point
    # point_a: a point
    # point_b: a point
    # output:
    # vec_db, a vector, such that d in ca, and db perpendicular to ca, and cd + db = cb
    @classmethod
    def vertical_line(
        cls, 
        point_c: np.ndarray, 
        point_a: np.ndarray, 
        point_b: np.ndarray
     ) -> np.ndarray:
        
        vec_ca = point_a - point_c
        vec_cb = point_b - point_c
        vec_cd = Helper.projection(vec_ca, vec_cb)
        vec_db = vec_cb - vec_cd
        
        return vec_db
    
    # 返回 point_p 沿着垂直 direction 的平面旋转 theta 弧度得到的点 point_p1
    @classmethod
    def rotate_one(
        cls,
        point_p: np.ndarray, 
        direction: np.ndarray, 
        theta: float
    ) -> np.ndarray:

        projection = direction * np.inner(direction, point_p) / np.inner(direction, direction)
        vec_dp = point_p - projection
        dir_j = np.cross(vec_dp, direction)
        dir_j = dir_j / np.linalg.norm(dir_j)
        dir_i = vec_dp / np.linalg.norm(vec_dp)
        length_dp = np.linalg.norm(vec_dp)
        vec_dp1 = length_dp * (cos(theta) * dir_i + sin(theta) * dir_j)
        vec_pp1 = vec_dp1 - vec_dp
        point_p1 = point_p + vec_pp1
        
        return point_p1
    
    # input:
    # direction: direction of the rotate axis
    # points_p: points to be rotate
    # theta: angle size of rotate
    @classmethod
    def rotate_many(
        cls,
        points_p: np.ndarray,
        direction: np.ndarray,
        theta: float
    ) -> np.ndarray:

        inner_ab = np.sum((direction.reshape(1,3)) * points_p, axis=1)
        inner_aa = np.inner(direction, direction)
        projections = direction.reshape(1, 3) * (inner_ab / inner_aa).reshape((inner_ab.shape[0], 1))
        vecs_dp = points_p - projections
        dirs_j = np.cross(direction, vecs_dp, axisb=1)
        dirs_j = dirs_j / np.linalg.norm(dirs_j, axis=1).reshape(dirs_j.shape[0],1)
        dirs_i = vecs_dp / np.linalg.norm(vecs_dp, axis=1).reshape(vecs_dp.shape[0],1)
        lengths_dp = np.linalg.norm(vecs_dp, axis=1).reshape(points_p.shape[0], 1)
        vecs_dp1 = lengths_dp * (cos(theta) * dirs_i + sin(theta) * dirs_j)
        vecs_pp1 = vecs_dp1 - vecs_dp
        points_p1 = points_p + vecs_pp1

        return points_p1
                
    
    # input: 
    # v: length 3 vector, direction of the ray
    # triangle: 3 x 3 array, each row correspond to one point of a triangle
    # output:
    # True if v is perpendicular to triangle
    @classmethod
    def check_perpendicular(
        cls,
        v: np.ndarray,
        triangle: np.ndarray
    ) -> bool:
        point_a = triangle[0, :]
        point_b = triangle[1, :]
        point_c = triangle[2, :]
        vec_ab = point_b - point_a
        vec_ac = point_c - point_a
        vec_p = np.cross(vec_ab, vec_ac)
        eps = 1e-6
        if abs(Helper.cosine(vec_p, v) - 0.0) < eps:
            # now the ray is perpendicular to that plane
            return True
        else:
            # not perpendicular
            return False
    
    # input:
    # 
    # point_o: origin point
    # v: direction of ray
    #
    # point_a: base point of geometry object
    # point_b: second point of geometry object 
    # point_c: third point of geometry object
    #
    # output:
    # 
    # If the ray is not perpendicular to the plane where the geometry object reside in,
    # then, point_p, which is the intersection of the ray and the plane, can be written as:
    #
    # point_p = point_a + x * vec_ab + y * vec_ac,
    #
    # where vec_ab == point_b - point_a, vec_ac = point_c - point_a.
    #
    # Also, there must be some z, such that
    #   
    # point_p == point_o + z * v
    #
    # It return a column vector (z, x, y)^T.
    @classmethod
    def find_intersect(
        cls,
        point_o: np.ndarray,
        v: np.ndarray,
        point_a: np.ndarray,
        point_b: np.ndarray,
        point_c: np.ndarray
    ) -> np.ndarray:
        
        vec_ab = point_b - point_a
        vec_ac = point_c - point_a
        # solve the equation:
        # point_o + z * v == point_a + x * vec_ab + y * vec_ac
        # where z, x, y are unknows.
        c1 = (-1) * v.reshape(3, 1)
        c2 = vec_ab.reshape(3, 1)
        c3 = vec_ac.reshape(3, 1)
        coeff_A = np.concatenate(
            (c1, c2, c3,),
            axis=1
        )
        coeff_B = (point_o - point_a).reshape(3, 1)
        answer: np.ndarray = np.linalg.solve(coeff_A, coeff_B)
        # now answer[0] should be z, answer[1] should be x, and answer[2] should be y.

        return answer
    
    # input:
    # data: data[(i*n_rows):((i+1)*n_rows), :] for (i+1) th matrix A_i
    # output:
    # diag{ A_0, A_1, ..., A_m }
    @classmethod
    def make_block_diagonal(
        cls, 
        data: np.ndarray, 
        n_rows: int, 
    ) -> csr_matrix:
        n_cols = data.shape[1]
        n_matrices = floor(data.shape[0]/n_rows)
        if n_matrices == 0:
            data = [np.nan]
            indptr = [0,1]
            indices = [0]
            return csr_matrix((data, indices, indptr))
        
        indptr = np.arange(0, n_matrices*n_rows*n_cols + 1, n_cols)
        indices = np.arange(0, n_matrices*n_cols)
        indices = np.reshape(indices, newshape=(n_matrices, n_cols,))
        indices = np.repeat(indices, n_rows, axis=0)
        indices = np.reshape(indices, newshape=(indices.shape[0]*indices.shape[1],))
        data = np.reshape(data, newshape=(data.shape[0]*data.shape[1],))

        return csr_matrix((data, indices, indptr,))
