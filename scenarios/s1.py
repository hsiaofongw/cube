from canvas import Canvas
from cube import Cube
import numpy as np

class Scenario1:

    def get(self):

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
        img_width = 80
        img_height = 60
        pixels = canvas.get_pixel_points(
            angles_h,
            angles_v,
            img_width,
            img_height
        )

        camera = np.atleast_2d(camera)
        view_directions = pixels - camera

        points_a, points_b, points_c = cube.get_facades()

        return (
            camera,
            view_directions,
            points_a,
            points_b,
            points_c,
            pixels,
            img_width,
            img_height
        )