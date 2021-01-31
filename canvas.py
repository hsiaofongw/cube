import numpy as np
from helper import Helper
from math import tan
from typing import Tuple
import json
     
class Canvas:
    
    # create a camera and a canvas, 
    # a canvas is a 2-D thing use to paint 3-D objects
    def __init__(
        self, 
        camera: np.ndarray,
        view_direction: np.ndarray,
        canvas_distance: float,
    ):
        self.camera = camera
        view_direction = view_direction/np.linalg.norm(view_direction)
        self.view_direction = view_direction
        self.canvas_distance = canvas_distance
        
        self.canvas_center = camera + canvas_distance * view_direction
    
    # define the horizon for canvas, horizon should not parallel to the norm of canvas
    def set_horizon(self, horizon: np.ndarray) -> None:

        projection = self.view_direction  * np.inner(
            self.view_direction, horizon
        ) / np.inner(self.view_direction, self.view_direction)

        horizon = horizon - projection
        self.dir_x = horizon / np.linalg.norm(horizon)

        self.dir_y = np.cross(self.dir_x, self.view_direction)
        self.dir_y = self.dir_y / np.linalg.norm(self.dir_y)
    
    def set_angles(self, angles_h: np.ndarray, angles_v: np.ndarray) -> None:
        self.angles_h = angles_h
        self.angles_v = angles_v

    def set_pixels_size(self, width_pixels: int, height_pixels: int) -> None:
        self.width_pixels = width_pixels
        self.height_pixels = height_pixels
    
    def get_corners(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        angles_h = self.angles_h
        angles_v = self.angles_v

        left_arm = tan(angles_h[0]) * self.canvas_distance * self.dir_x
        right_arm = tan(angles_h[1]) * self.canvas_distance * self.dir_x

        top_arm = tan(angles_v[1]) * self.canvas_distance * self.dir_y
        bottom_arm = tan(angles_v[0]) * self.canvas_distance * self.dir_y

        lu_corner = self.canvas_center + left_arm + top_arm
        lb_corner = self.canvas_center + left_arm + bottom_arm
        rb_corner = self.canvas_center + right_arm + bottom_arm
        ru_corner = self.canvas_center + right_arm + top_arm

        return (
            lu_corner,
            lb_corner,
            rb_corner,
            ru_corner
        )
    
    # get locations of each canvas pixel 
    def get_pixel_points(self) -> np.ndarray:

        lu_corner, lb_corner, rb_corner, ru_corner = self.get_corners()
        width_pixels, height_pixels = self.width_pixels, self.height_pixels
        
        baby_step_x = (ru_corner - lu_corner) / width_pixels
        baby_step_y = (lb_corner - lu_corner) / height_pixels

        baby_step_x = np.atleast_2d(baby_step_x)
        baby_step_y = np.atleast_2d(baby_step_y)

        x_steps, y_steps = np.meshgrid(
            np.arange(width_pixels),
            np.arange(height_pixels)
        )

        x_steps = x_steps.flatten(order='C')
        y_steps = y_steps.flatten(order='C')

        x_steps = np.atleast_2d(x_steps).T
        y_steps = np.atleast_2d(y_steps).T

        pixels = lu_corner + x_steps * baby_step_x + y_steps * baby_step_y

        return pixels
        
    def export_to_object(self) -> dict:

        lu, lb, rb, ru = self.get_corners()

        info = {
            "camera": self.camera.tolist(),
            "corners": {
                "lu": lu.tolist(),
                "lb": lb.tolist(),
                "rb": rb.tolist(),
                "ru": ru.tolist()
            },
            "size": {
                "width": self.width_pixels,
                "height": self.height_pixels
            }
        }
        
        return info
    
    def export_to_json_file(self, json_filename: str) -> None:

        with open(json_filename, 'w') as f:
            json.dump(self.export_to_object(), f)