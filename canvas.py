import numpy as np
from helper import Helper
from math import tan
     
class Canvas:
    
    def __init__(
        self, 
        camera: np.ndarray,
        view_direction: np.ndarray,
        canvas_distance: float,
        canvas_base_line: np.ndarray,
        view_angles_h: np.ndarray,
        view_angles_v: np.ndarray,
        canvas_width_pixels: int,
        canvas_height_pixels: int
    ):
        self.camera = camera
        self.view_direction = view_direction
        self.view_direction_norm = view_direction/np.linalg.norm(view_direction)
        self.view_angles_h = view_angles_h
        self.view_angles_v = view_angles_v
        self.canvas_distance = canvas_distance
        
        self.canvas_center = camera + canvas_distance * self.view_direction_norm
        self.canvas_norm = self.view_direction_norm
        self.canvas_width_pixels = canvas_width_pixels
        self.canvas_height_pixels = canvas_height_pixels
        
        cosine_of_canvas_base_line =  Helper.cosine(canvas_base_line, self.canvas_norm)
        self.canvas_base_line += cosine_of_canvas_base_line * self.canvas_norm
        
        if abs(Helper.cosine(canvas_base_line, self.canvas_norm) - 1) > 1e-4:
            self.canvas_base_line += (-2) * cosine_of_canvas_base_line * self.canvas_norm
        
    def get_pixel_points(self) -> np.ndarray:

        canvas_base_i = self.canvas_base_line
        canvas_base_j = np.cross(canvas_base_i, self.canvas_norm)
        canvas_base_i = canvas_base_i/np.linalg.norm(canvas_base_i)
        canvas_base_j = canvas_base_j/np.linalg.norm(canvas_base_j)

        canvas_width_pixels = self.canvas_width_pixels
        canvas_height_pixels = self.canvas_height_pixels

        canvas_left_arm = tan(self.view_angles_h[0]) * self.canvas_distance * canvas_base_i
        canvas_right_arm = tan(self.view_angles_h[1]) * self.canvas_distance * canvas_base_i
        canvas_top_arm = tan(self.view_angles_v[1]) * self.canvas_distance * canvas_base_j
        canvas_bottom_arm = tan(self.view_angles_v[0]) * self.canvas_distance * canvas_base_j

        canvas_top_left_vertex = self.canvas_center + canvas_left_arm + canvas_top_arm
        canvas_top_right_vertex = self.canvas_center + canvas_right_arm + canvas_top_arm
        canvas_bottom_left_vertex = self.canvas_center + canvas_left_arm + canvas_bottom_arm

        baby_step_i = (canvas_top_right_vertex - canvas_top_left_vertex) / canvas_width_pixels
        baby_step_j = (canvas_bottom_left_vertex - canvas_top_left_vertex) / canvas_height_pixels

        canvas_pixel_points = np.zeros((canvas_width_pixels * canvas_height_pixels, 3,), dtype = np.float)
        for row in range(canvas_width_pixels):
            for col in range(canvas_height_pixels):
                i = row * canvas_height_pixels + col
                canvas_pixel_points[i, :] = canvas_top_left_vertex + col * baby_step_j + row * baby_step_i

        self.canvas_pixel_points = canvas_pixel_points
    
    def get_camera_point(self) -> np.ndarray:
        return self.camera
