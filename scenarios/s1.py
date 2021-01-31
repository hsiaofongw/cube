from canvas import Canvas
from cube import Cube
import numpy as np
from typing import Tuple

class Scenario1:

    def get(self) -> Tuple[Canvas, np.ndarray]:

        cube_center = np.array([10, 0, 0])
        edge_length = 3.2
        cube = Cube(cube_center, edge_length)
        cube.rotate(
            np.array([0, 0, 1]),
            cube.cube_center,
            np.radians(15)
        )

        camera = np.array([0, 2, 4])
        canvas_distance = 3
        view_direction = cube_center - camera

        canvas = Canvas(
            camera,
            view_direction,
            canvas_distance
        )

        horizon = np.array([0, -1, 0])
        canvas.set_horizon(horizon)

        angles_h = np.radians(np.array([-29, 29]))
        angles_v = np.radians(np.array([-25, 25]))
        canvas.set_angles(angles_h, angles_v)

        img_width = 600
        img_height = 400
        canvas.set_pixels_size(img_width, img_height)

        triangles = cube.get_triangles()

        return (canvas, triangles,)
