import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cosine as cosine_distance
from helper import Helper
from typing import Tuple

class Cube:
    
    # 坐标系：
    #
    #     k    j
    #     |   /      
    #     |  /     
    #     | /   
    #     |/  
    #     + - - - - - i
    #
    # 初始化一个中心在 cube_center, 棱长为 edge_length 的正方体
    def __init__(
        self,
        cube_center: np.ndarray,
        edge_length: float
    ) -> None:
        
        self.cube_center = cube_center.astype(np.float)
        self.edge_length = edge_length
        
        vec_i = np.array([1, 0, 0], dtype=np.float)
        vec_j = np.array([0, 1, 0], dtype=np.float)
        vec_k = np.array([0, 0, 1], dtype=np.float)
        
        h_vec_i = 0.5 * edge_length * vec_i
        h_vec_j = 0.5 * edge_length * vec_j
        h_vec_k = 0.5 * edge_length * vec_k
        
        a0 = (cube_center + h_vec_i - h_vec_j - h_vec_k).reshape(1, 3)
        b0 = (cube_center - h_vec_i - h_vec_j - h_vec_k).reshape(1, 3)
        c0 = (cube_center - h_vec_i + h_vec_j - h_vec_k).reshape(1, 3)
        d0 = (cube_center + h_vec_i + h_vec_j - h_vec_k).reshape(1, 3)
        
        a1 = a0 + edge_length * vec_k
        b1 = b0 + edge_length * vec_k
        c1 = c0 + edge_length * vec_k
        d1 = d0 + edge_length * vec_k
        
        self.cube_vertices = np.concatenate(
            (a0,b0,c0,d0,a1,b1,c1,d1,),
            axis=0
        )
        
    def move(self, replacement: np.ndarray) -> None:
        self.cube_vertices += replacement
        self.cube_center += replacement
        
    def rotate(
        self, 
        rotate_axis: np.ndarray,
        center: np.ndarray,
        radian: float
    ) -> None:
        self.cube_vertices = Helper.rotate_many(self.cube_vertices, rotate_axis, center, radian)
    
    def scale(
        self,
        factor: float,
    ) -> None:
        center = np.reshape(self.cube_center, newshape=(1,3,))
        extends = self.cube_vertices - center
        extends = extends * factor
        self.cube_vertices = center + extends
    
    def get_facades_indexes(self) -> np.ndarray:
        
        return np.array([
            [1, 3, 2],
            [1, 3, 4],
            
            [5, 7, 6],
            [5, 7, 8],
            
            [6, 3, 2],
            [6, 3, 7],
            
            [5, 4, 1],
            [5, 4, 8],
            
            [1, 6, 2],
            [1, 6, 5],
            
            [7, 4, 3],
            [7, 4, 8]
        ]) - 1
    
    def get_triangles(self) -> np.ndarray:

        vertexes = self.cube_vertices
        triangle_indexes = self.get_facades_indexes()
        points_a = np.zeros(shape=(12, 3,), dtype=np.float)
        points_b = np.zeros(shape=(12, 3,), dtype=np.float)
        points_c = np.zeros(shape=(12, 3,), dtype=np.float)
        for i in range(triangle_indexes.shape[0]):
            point_a_ind = triangle_indexes[i, 0]
            point_b_ind = triangle_indexes[i, 1]
            point_c_ind = triangle_indexes[i, 2]

            points_a[i, :] = vertexes[point_a_ind, :]
            points_b[i, :] = vertexes[point_b_ind, :]
            points_c[i, :] = vertexes[point_c_ind, :]

        return np.concatenate(
            (points_a, points_b, points_c,),
            axis=1
        )
    
    def get_edge_lengths(self) -> np.ndarray:
        edge_lengths = np.zeros(shape=(12,), dtype=np.float)

        edge_lengths[0] = np.linalg.norm(self.cube_vertices[0] - self.cube_vertices[1])
        edge_lengths[1] = np.linalg.norm(self.cube_vertices[1] - self.cube_vertices[2])
        edge_lengths[2] = np.linalg.norm(self.cube_vertices[2] - self.cube_vertices[3])
        edge_lengths[3] = np.linalg.norm(self.cube_vertices[3] - self.cube_vertices[0])

        edge_lengths[4] = np.linalg.norm(self.cube_vertices[4] - self.cube_vertices[5])
        edge_lengths[5] = np.linalg.norm(self.cube_vertices[5] - self.cube_vertices[6])
        edge_lengths[6] = np.linalg.norm(self.cube_vertices[6] - self.cube_vertices[7])
        edge_lengths[7] = np.linalg.norm(self.cube_vertices[7] - self.cube_vertices[4])

        edge_lengths[8] = np.linalg.norm(self.cube_vertices[0] - self.cube_vertices[4])
        edge_lengths[9] = np.linalg.norm(self.cube_vertices[1] - self.cube_vertices[5])
        edge_lengths[10] = np.linalg.norm(self.cube_vertices[2] - self.cube_vertices[6])
        edge_lengths[11] = np.linalg.norm(self.cube_vertices[3] - self.cube_vertices[7])

        return edge_lengths
   