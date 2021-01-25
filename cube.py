import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cosine as cosine_distance
from helper import Helper

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
        
        self.cube_center = cube_center
        self.edge_length = edge_length
        
        vec_i = np.array([1, 0, 0])
        vec_j = np.array([0, 1, 0])
        vec_k = np.array([0, 0, 1])
        
        h_vec_i = 0.5 * edge_length * vec_i
        h_vec_j = 0.5 * edge_length * vec_j
        h_vec_k = 0.5 * edge_length * vec_k
        
        a0 = (cube_center + h_vec_i - h_vec_j - h_vec_k).reshape(1, 3)
        b0 = (cube_center - h_vec_i - h_vec_j - h_vec_k).reshape(1, 3)
        c0 = (cube_center - h_vec_i + h_vec_j - h_vec_k).reshape(1, 3)
        d0 = (cube_center + h_vec_i + h_vec_j - h_vec_k).reshape(1, 3)
        
        a1 = a0 + 2 * vec_k
        b1 = b0 + 2 * vec_k
        c1 = c0 + 2 * vec_k
        d1 = d0 + 2 * vec_k
        
        self.cube_vertices = np.concatenate(
            (a0,b0,c0,d0,a1,b1,c1,d1,),
            axis=0
        )
        
        self.vec_i = vec_i
        self.vec_j = vec_j
        self.vec_k = vec_k
    
    def move(self, replacement: np.ndarray) -> None:
        self.cube_vertices += replacement
        
    def rotate(
        self, 
        rotate_axis: np.ndarray,
        rotate_center: np.ndarray,
        degree: float = 0,
        radian: float = 0
    ) -> None:
        
        self.cube_vertices = Helper.rotate_many(self.cube_vertices, rotate_center, rotate_axis, radian)
    
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
    
    def get_facades(self) -> np.ndarray:

        vertexes = self.cube_vertices
        triangles = np.zeros(shape = (12, 3, 3,), dtype=np.float)
        triangle_indexes = self.get_facades_indexes()
        for i in range(triangle_indexes.shape[0]):
            point_a_ind = triangle_indexes[i, 0]
            point_b_ind = triangle_indexes[i, 1]
            point_c_ind = triangle_indexes[i, 2]

            point_a = vertexes[point_a_ind, :]
            point_b = vertexes[point_b_ind, :]
            point_c = vertexes[point_c_ind, :]

            triangles[i][0] = point_a
            triangles[i][1] = point_b
            triangles[i][2] = point_c
        
        return triangles
   


points = np.random.rand(100, 3)
center = np.random.rand(3)
axis =  np.random.rand(3)

norms = np.linalg.norm(points, axis=1)
selector = norms > 1e-6



print(norms.shape)