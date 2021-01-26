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
        
        vec_dp = Helper.vertical_line(point_c, point_t, point_p)
        vec_i = vec_dp / np.linalg.norm(vec_dp)
        vec_ct = point_t - point_c
        vec_j = np.cross(vec_i, vec_ct)
        vec_j = vec_j / np.linalg.norm(vec_j)
        vec_dp1 = np.linalg.norm(vec_dp) * (cos(theta) * vec_i + sin(theta) * vec_j)
        point_p1 = point_p - vec_dp + vec_dp1
        
        return (vec_dp1, point_p1, )
    
    # input:
    # 
    # point_o: origin point
    # v: direction of ray
    #
    # point_a: base point of geometry object
    # point_b: second point of geometry object 
    # point_c: third point of geometry object
    #
    # mat_Aeq: a matrix
    # mat_Beq: a matrix
    #
    # mat_Aleq: a matrix
    # mat_Bleq: a matrix
    #
    # mat_Ale: a matrix
    # mat_Ble: a matrix
    # 
    # output:
    # 
    # True if a ray originate from point_o, by the direction of v, has 
    # an intersection point inside geometry object point_a - point_b - point_c
    # 
    # If the ray is not perpendicular to the plane where the geometry object reside in,
    # then, point_p, which is the intersection of the ray and the plane, can be written as:
    #
    # point_p = point_a + x * vec_ab + y * vec_ac,
    #
    # where vec_ab == point_b - point_a, vec_ac = point_c - point_a.
    #
    # Therefore, let point_p_arg be a column vector (x, y)^T, then, point_p is 'inside' that area
    # can be explicitly defined as:
    #
    #     mat_Aeq @ point_p_arg == mat_Beq
    # and mat_Aleq @ point_p_arg <= mat_Bleq
    # and mat_Ale @ point_p_arg < mat_Ble
    #
    # where the at sign '@' denote matrix multiplication.
    @classmethod
    def check_intersect(
        cls,
        point_o: np.ndarray,
        v: np.ndarray,
        point_a: np.ndarray,
        point_b: np.ndarray,
        point_c: np.ndarray,
        mat_Aeq: np.ndarray,
        mat_Beq: np.ndarray,
        mat_Aleq: np.ndarray,
        mat_Bleq: np.ndarray,
        mat_Ale: np.ndarray,
        mat_Ble: np.ndarray
    ) -> bool:
        vec_ab = point_b - point_a
        vec_ac = point_c - point_a
        vec_p = np.cross(vec_ab, vec_ac)
        eps = 1e-6
        if abs(Helper.cosine(vec_p, v) - 0.0) < eps:
            # now the ray is perpendicular to that plane, so there is no intersection.
            return False
        
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
        answer = np.linalg.solve(coeff_A, coeff_B)
        # now answer[0] should be z, answer[1] should be x, and answer[2] should be y.

        print(answer)

        return True


point_o = np.random.rand(3)
v = np.random.rand(3)
point_a = np.random.rand(3)
point_b = np.random.rand(3)
point_c = np.random.rand(3)

mat_Aeq = np.zeros(1)
mat_Beq = np.zeros(1)
mat_Aleq = np.zeros(1)
mat_Bleq = np.zeros(1)
mat_Ale = np.zeros(1)
mat_Ble = np.zeros(1)

Helper.check_intersect(
    point_o, 
    v, 
    point_a, 
    point_b, 
    point_c, 
    mat_Aeq, 
    mat_Beq, 
    mat_Aleq, 
    mat_Bleq, 
    mat_Ale, 
    mat_Ble
)
