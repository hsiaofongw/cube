from canvas import Canvas
from cube import Cube
import numpy as np
from helper import Helper
from tqdm import tqdm
from scipy.sparse.linalg import splu
import time

class SimpleRenderer:

    def render(
        self,
        camera: np.ndarray,
        pixel_points: np.ndarray,
        points_a: np.ndarray,
        points_b: np.ndarray,
        points_c: np.ndarray,
    ):
        camera = np.atleast_2d(camera)
        view_directions = pixel_points - camera

        image = np.zeros(
            shape=(pixel_points.shape[0],),
            dtype=np.int
        )

        vecs_ab = points_b - points_a
        vecs_ac = points_c - points_a
        camera = camera.T
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
        
        return image


class CPUMatrixRenderer:

    def render(
        self,
        camera: np.ndarray,
        pixels: np.ndarray,
        points_a: np.ndarray,
        points_b: np.ndarray,
        points_c: np.ndarray
    ) -> np.ndarray:

        print("constructing coefficients matrices ...")

        n_triangles = points_a.shape[0]
        n_pixels = pixels.shape[0]

        vecs_ab = points_b - points_a
        vecs_ac = points_c - points_a

        views = pixels - camera
        views = np.repeat(views, repeats=n_triangles, axis=0)

        points_a = np.tile(points_a, reps=(n_pixels, 1,))
        vecs_ab = np.tile(vecs_ab, reps=(n_pixels, 1,))
        vecs_ac = np.tile(vecs_ac, reps=(n_pixels, 1,))

        coeff_B = np.atleast_2d(camera) - points_a
        coeff_B = coeff_B.flatten(order='C')

        views = np.atleast_2d(views.flatten(order='C')).T
        vecs_ab = np.atleast_2d(vecs_ab.flatten(order='C')).T
        vecs_ac = np.atleast_2d(vecs_ac.flatten(order='C')).T
        coeff_A = np.concatenate(
            (-views, vecs_ab, vecs_ac,),
            axis=1
        )

        coeff_A = Helper.make_block_diagonal(coeff_A, 3)

        coeff_A = coeff_A.tocsc()

        print(f"coeff_A: {coeff_A.shape}")
        print(f"coeff_B: {coeff_B.shape}")

        t_start = time.time()
        print(f"{t_start}: start solving ...")
        solver = splu(coeff_A)
        solution = solver.solve(coeff_B)
        t_finish = time.time()
        print(f"{t_finish}: solved.")
        t_delta = t_finish - t_start
        print(f"time consume: {t_delta}")

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

