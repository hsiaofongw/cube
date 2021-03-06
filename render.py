from canvas import Canvas
import numpy as np
from helper import Helper
from tqdm import tqdm
import time
from solver import LUMultipleEquationsSolver

class SimpleIntegrator:

    def integrate(
        self, 
        solution: np.ndarray,
        n_pixels: int,
        n_triangles: int
    ) -> np.ndarray:

        solution = solution.reshape(n_pixels * n_triangles, 3)

        zs = solution[:, 0]
        xs = solution[:, 1]
        ys = solution[:, 2]

        zs = zs.reshape(n_pixels, n_triangles)
        xs = xs.reshape(n_pixels, n_triangles)
        ys = ys.reshape(n_pixels, n_triangles)

        selectors = np.logical_and(
            zs > 0,
            np.logical_and(
                xs + ys <= 1,
                np.logical_and(
                    xs >= 0,
                    ys >= 0
                )
            )
        )
        selectors = np.any(selectors, axis=1)

        image = np.zeros(
            shape=(n_pixels, 1,),
            dtype=np.int
        )
        image[selectors] = 1

        return image

class SimpleRenderer:

    def render(
        self,
        canvas: Canvas,
        triangles: np.ndarray
    ):
        camera = canvas.camera
        pixel_points = canvas.get_pixel_points()
        view_directions = pixel_points - camera

        points_a: np.ndarray = triangles[:, 0:3]
        points_b: np.ndarray = triangles[:, 3:6]
        points_c: np.ndarray = triangles[:, 6:9]

        image = np.zeros(
            shape=(pixel_points.shape[0],),
            dtype=np.int
        )

        vecs_ab = points_b - points_a
        vecs_ac = points_c - points_a
        camera = camera.T

        t_start = time.time()
        print(f"{t_start}: start rendering ...")
        for k in tqdm(range(pixel_points.shape[0])):
            for l in range(points_a.shape[0]):
                v = view_directions[k:(k+1), :].T
                point_a = points_a[l:(l+1), :].T
                vec_ab = vecs_ab[l:(l+1), :].T
                vec_ac = vecs_ac[l:(l+1), :].T

                coeff_A = np.concatenate(
                    (-v, vec_ab, vec_ac,),
                    axis=1
                )

                coeff_B = camera - point_a

                sol = np.linalg.solve(coeff_A, coeff_B)
                z = sol[0, 0]
                x = sol[1, 0]
                y = sol[2, 0]

                if (z > 0) and (x + y <= 1) and (x >= 0) and (y >= 0):
                    image[k] = 1
                    break
        
        t_finish = time.time()
        t_delta = t_finish - t_start
        print(f"{t_finish}: render completed.")
        print(f"time consume: {t_delta}")
        
        return image


class CPUMatrixRenderer:

    def render(
        self,
        canvas: Canvas,
        triangles: np.ndarray
    ) -> np.ndarray:

        points_a = triangles[:, 0:3]
        points_b = triangles[:, 3:6]
        points_c = triangles[:, 6:9]

        pixels = canvas.get_pixel_points()

        coeff_A, coeff_B = Helper.make_coefficients(
            canvas.camera,
            pixels,
            points_a,
            points_b,
            points_c
        )

        t_start = time.time()
        print(f"{t_start}: rendering ...")

        solver = LUMultipleEquationsSolver()
        solution = solver.solve(coeff_A, coeff_B)

        t_finish = time.time()
        print(f"{t_finish}: done.")
        t_delta = t_finish - t_start
        print(f"time consume: {t_delta}")
        
        n_pixels = pixels.shape[0]
        n_triangles = triangles.shape[0]
        integrator = SimpleIntegrator()
        image = integrator.integrate(
            solution, n_pixels, n_triangles
        )

        return image
