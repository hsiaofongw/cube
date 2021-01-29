from canvas import Canvas
from cube import Cube
import numpy as np
from helper import Helper
from tqdm import tqdm

class SimpleRenderer:

    def render(
        self,
        camera: np.ndarray,
        pixel_points: np.ndarray,
        points_a: np.ndarray,
        points_b: np.ndarray,
        points_c: np.ndarray,
        img_width: int,
        img_height: int
    ):

        camera = np.atleast_2d(camera)
        view_directions = pixel_points - camera

        image = np.zeros(
            shape = (img_height, img_width,), 
            dtype = np.int
        )

        vecs_ab = points_b - points_a
        vecs_ac = points_c - points_a
        triangle_norms = np.cross(
            vecs_ab,
            vecs_ac,
            axisa=1,
            axisb=1
        )

        cosines = np.abs(Helper.cosine_many_to_many(
            view_directions,
            triangle_norms
        ))

        criterion = 1e-10
        camera = camera.T

        for i in tqdm(range(image.shape[0])):
            for j in range(image.shape[1]):
                k = i * img_width + j
                pixel = 0
                for l in range(points_a.shape[0]):

                    cosine = cosines[k, l]
                    if cosine < criterion:
                        continue

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

                    if (x + y <= 1) and (x >= 0) and (y >= 0):
                        pixel = pixel + 1
                        break
                
                if pixel >= 1:
                    image[i, j] = 1
        
        return image
