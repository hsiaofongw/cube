import numpy as np
from scipy.spatial.transform import Rotation as R
from math import sin, cos
from scipy.spatial.distance import cosine as cosine_distance
from typing import Tuple

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
    
    # input:
    # point_c: center of the sphere
    # point_t: vec_ct denote sphere direction axis
    # point_p: point to be rotate
    # theta: radian of rotate
    # output:
    # vec_dp1: satisfy that cosine(vec_dp1, vec_dp) = cosine(theta) and 
    # cosine(vec_dp1, vec_ct) == 0, 
    # where point_d satisfy that cosine(vec_dp, vec_ct) == 0,
    # point_p1: p1 in vec_dp1
    @classmethod
    def rotate_one(
        cls,
        point_c: np.ndarray,
        point_t: np.ndarray,
        point_p: np.ndarray,
        theta: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        vec_cp = point_p - point_c
        vec_dp = Helper.vertical_line(point_c, point_t, point_p)
        vec_i = vec_dp / np.linalg.norm(vec_dp)
        vec_ct = point_t - point_c
        vec_j = np.cross(vec_i, vec_ct)
        vec_j = vec_j / np.linalg.norm(vec_j)
        vec_dp1 = np.linalg.norm(vec_dp) * (cos(theta) * vec_i + sin(theta) * vec_j)
        point_p1 = point_p - vec_dp + vec_dp1
        
        return (vec_dp1, point_p1, )

